"""
build_features.py  -  Phase 6 (v3): leakage-safe features, one table per cohort

Every feature is computed ONLY from transactions dated on or before the cohort's as-of date
(Feb cohort: 2017-01-31, Mar cohort: 2017-02-28).  Nothing from the label window is touched.

Two feature sets (both get reported in Phase 7):
  SET A  = static member attributes + transaction HISTORY aggregates and ratios.
           Deliberately excludes the user's *current* membership snapshot.
  SET B  = A + membership STATE as of the as-of date (the latest transaction's plan, auto-renew flag,
           cancel flag, and days until its expiry).  Legitimately knowable on the as-of date, but
           near-label in strength, which is why published "near-label removed" results fall to 0.72-0.85.

Columns from dim_member that must NEVER become features (they use the label window) - asserted absent:
  last_expiry_full, window_visible, is_window_censored, reconstructed_churn, is_churn_supplied

Gates (any failure stops the script):
  1. one row per cohort user, msno unique, row count equals dim_member
  2. label rate matches the published figure (Feb 6.39%, Mar 8.99%)
  3. latest transaction date used <= as-of date
  4. no forbidden column among the features
  5. n_tx reconciles with dim_member.n_transactions_asof (Feb asserted; Mar reported - dim did not
     de-duplicate the v1+v2 union, this script does)
  6. leakage tripwire: no single numeric feature may reach AUC 0.90 on its own

Reads : data/core/dim_member_{feb,mar}.parquet, data/staging/transactions_v1.parquet, data/staging/transactions.parquet
Writes: data/core/features_feb.parquet, data/core/features_mar.parquet, data/core/feature_sets.json
"""
import json
import os

import duckdb
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")

GAP_DAYS = 30
EXPIRY_CLIP = 400          # days_until_expiry clipped to +/- this (far-future junk expiries exist, e.g. year 2036)
AUC_TRIPWIRE = 0.90

SET_A_STATIC = ["has_member_record", "city", "registered_via", "age_clean", "gender_clean",
                "tenure_days_asof", "registration_after_asof"]
SET_A_HISTORY = ["has_pre_asof_transactions", "n_tx", "n_cancel", "cancel_rate", "auto_renew_rate",
                 "n_autorenew_switch_offs", "n_prior_lapses", "avg_plan_days", "n_distinct_plan_days",
                 "n_distinct_pay_methods", "total_paid", "avg_paid", "free_share", "avg_discount_rate",
                 "days_since_first_tx", "days_since_last_tx", "first_payment_free", "first_plan_days"]
SET_B_EXTRA = ["last_plan_days", "last_amount_paid", "last_is_auto_renew", "last_is_cancel",
               "last_payment_method_id", "days_until_expiry"]
SET_A = SET_A_STATIC + SET_A_HISTORY
SET_B = SET_A + SET_B_EXTRA
CATEGORICAL = ["city", "registered_via", "gender_clean", "last_payment_method_id"]
FORBIDDEN = {"last_expiry_full", "window_visible", "is_window_censored", "reconstructed_churn", "is_churn_supplied"}

os.makedirs(P("data", "core"), exist_ok=True)
con = duckdb.connect()
con.execute(f"CREATE VIEW tx_v1 AS SELECT * FROM read_parquet('{P('data','staging','transactions_v1.parquet')}')")
con.execute(f"CREATE VIEW tx_v2 AS SELECT * FROM read_parquet('{P('data','staging','transactions.parquet')}')")

TX_COLS = ("msno, transaction_date, membership_expire_date, payment_method_id, payment_plan_days, "
           "plan_list_price, actual_amount_paid, is_auto_renew, is_cancel")


def build(label, dim_name, as_of, feature_tx_sql, out_name, expected_rate, assert_ntx):
    print("\n" + "=" * 78 + f"\nCOHORT: {label}  (as-of {as_of})\n" + "=" * 78)
    con.execute(f"CREATE OR REPLACE VIEW cohort_dim AS SELECT * FROM read_parquet('{P('data','core',dim_name+'.parquet')}')")

    # all transactions available on the as-of date, de-duplicated (Decision 7) and restricted to cohort users
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tx_f AS
        SELECT * FROM ({feature_tx_sql})
        WHERE msno IN (SELECT msno FROM cohort_dim)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY msno, transaction_date
                                   ORDER BY membership_expire_date DESC) = 1
    """)
    max_tx = con.execute("SELECT MAX(transaction_date) FROM tx_f").fetchone()[0]
    assert str(max_tx)[:10] <= as_of, f"transaction dated {max_tx} is after the as-of date {as_of} - LEAK"
    print(f"Latest transaction date used: {str(max_tx)[:10]}  (as-of {as_of})  OK")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tx_w AS
        SELECT *,
               LAG(is_auto_renew) OVER (PARTITION BY msno ORDER BY transaction_date) AS prev_auto,
               MAX(membership_expire_date) OVER (PARTITION BY msno ORDER BY transaction_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
        FROM tx_f
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE agg AS
        SELECT msno,
               COUNT(*)                                             AS n_tx,
               SUM(is_cancel)::BIGINT                               AS n_cancel,
               AVG(is_cancel::DOUBLE)                               AS cancel_rate,
               AVG(is_auto_renew::DOUBLE)                           AS auto_renew_rate,
               SUM((prev_auto = 1 AND is_auto_renew = 0)::INT)::BIGINT AS n_autorenew_switch_offs,
               SUM((prev_end IS NOT NULL AND transaction_date > prev_end + {GAP_DAYS})::INT)::BIGINT AS n_prior_lapses,
               AVG(payment_plan_days::DOUBLE)                       AS avg_plan_days,
               COUNT(DISTINCT payment_plan_days)                    AS n_distinct_plan_days,
               COUNT(DISTINCT payment_method_id)                    AS n_distinct_pay_methods,
               SUM(actual_amount_paid)::DOUBLE                      AS total_paid,
               AVG(actual_amount_paid::DOUBLE)                      AS avg_paid,
               AVG((actual_amount_paid = 0)::INT)                   AS free_share,
               AVG(CASE WHEN plan_list_price > 0
                        THEN (plan_list_price - actual_amount_paid)::DOUBLE / plan_list_price END) AS avg_discount_rate,
               DATE_DIFF('day', MIN(transaction_date), DATE '{as_of}') AS days_since_first_tx,
               DATE_DIFF('day', MAX(transaction_date), DATE '{as_of}') AS days_since_last_tx,
               (ARG_MIN(actual_amount_paid, transaction_date) = 0)::INT AS first_payment_free,
               ARG_MIN(payment_plan_days, transaction_date)         AS first_plan_days,
               -- SET B: state of the latest transaction on or before the as-of date
               ARG_MAX(payment_plan_days, transaction_date)         AS last_plan_days,
               ARG_MAX(actual_amount_paid, transaction_date)        AS last_amount_paid,
               ARG_MAX(is_auto_renew, transaction_date)             AS last_is_auto_renew,
               ARG_MAX(is_cancel, transaction_date)                 AS last_is_cancel,
               ARG_MAX(payment_method_id, transaction_date)         AS last_payment_method_id,
               DATE_DIFF('day', DATE '{as_of}', ARG_MAX(membership_expire_date, transaction_date)) AS days_until_expiry_raw
        FROM tx_w GROUP BY msno
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE feat AS
        SELECT d.msno,
               d.is_churn_supplied::INT AS is_churn,
               -- static member attributes (SET A)
               d.has_member_record::INT AS has_member_record, d.city, d.registered_via, d.age_clean,
               d.gender_clean, d.tenure_days_asof, d.registration_after_asof::INT AS registration_after_asof,
               -- history (SET A)
               d.has_pre_asof_transactions::INT AS has_pre_asof_transactions,
               COALESCE(a.n_tx, 0) AS n_tx, COALESCE(a.n_cancel, 0) AS n_cancel, a.cancel_rate, a.auto_renew_rate,
               COALESCE(a.n_autorenew_switch_offs, 0) AS n_autorenew_switch_offs,
               COALESCE(a.n_prior_lapses, 0) AS n_prior_lapses,
               a.avg_plan_days, a.n_distinct_plan_days, a.n_distinct_pay_methods, a.total_paid, a.avg_paid,
               a.free_share, a.avg_discount_rate, a.days_since_first_tx, a.days_since_last_tx,
               a.first_payment_free, a.first_plan_days,
               -- state as of the as-of date (SET B)
               a.last_plan_days, a.last_amount_paid, a.last_is_auto_renew, a.last_is_cancel,
               a.last_payment_method_id,
               LEAST(GREATEST(a.days_until_expiry_raw, -{EXPIRY_CLIP}), {EXPIRY_CLIP}) AS days_until_expiry,
               a.days_until_expiry_raw
        FROM cohort_dim d LEFT JOIN agg a ON a.msno = d.msno
    """)

    # ---------------------------------------------------------------- gates
    n_rows, n_users, n_dim = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT msno), (SELECT COUNT(*) FROM cohort_dim) FROM feat").fetchone()
    assert n_rows == n_users == n_dim, f"rows {n_rows:,} / unique {n_users:,} / dim {n_dim:,}"
    rate = con.execute("SELECT AVG(is_churn::DOUBLE) FROM feat").fetchone()[0]
    assert abs(rate - expected_rate) < 0.0005, f"label rate {rate:.4f} differs from published {expected_rate}"
    print(f"{n_rows:,} users, one row each; churn rate {100*rate:.2f}% (published {100*expected_rate:.2f}%)  OK")

    cols = [r[0] for r in con.execute("DESCRIBE feat").fetchall()]
    leaked = FORBIDDEN & (set(cols) - {"is_churn"})
    assert not leaked, f"forbidden columns in feature table: {leaked}"
    missing = [c for c in SET_B if c not in cols]
    assert not missing, f"declared features missing from the table: {missing}"

    mism = con.execute("""
        SELECT COUNT(*) FROM feat f JOIN cohort_dim d USING (msno)
        WHERE f.n_tx <> d.n_transactions_asof""").fetchone()[0]
    print(f"n_tx vs dim_member.n_transactions_asof: {mism:,} users differ")
    if assert_ntx:
        assert mism == 0, f"{mism:,} users where n_tx disagrees with dim_member"
    clipped = con.execute(f"SELECT COUNT(*) FROM feat WHERE ABS(days_until_expiry_raw) > {EXPIRY_CLIP}").fetchone()[0]
    print(f"days_until_expiry clipped to +/-{EXPIRY_CLIP} for {clipped:,} users")

    con.execute(f"COPY feat TO '{P('data','core',out_name+'.parquet')}' (FORMAT PARQUET)")
    print(f"Wrote {P('data','core',out_name+'.parquet')}")
    return con.execute("SELECT * FROM feat").df()


feb = build("Feb (train.csv)", "dim_member_feb", "2017-01-31",
            f"SELECT {TX_COLS} FROM tx_v1 WHERE transaction_date <= DATE '2017-01-31'",
            "features_feb", 0.0639, True)
mar = build("Mar (train_v2.csv)", "dim_member_mar", "2017-02-28",
            f"""SELECT {TX_COLS} FROM tx_v1 WHERE transaction_date <= DATE '2017-02-28'
                UNION ALL
                SELECT {TX_COLS} FROM tx_v2 WHERE transaction_date <= DATE '2017-02-28'""",
            "features_mar", 0.0899, False)

# ---------------------------------------------------------------- leakage tripwire: one feature at a time
numeric = [c for c in SET_B if c not in CATEGORICAL]
print("\n" + "=" * 78 + "\nLEAKAGE TRIPWIRE - single-feature AUC (direction-free; missing -> -1e9)\n" + "=" * 78)
tab = {}
for name, df in (("Feb", feb), ("Mar", mar)):
    y = df["is_churn"].to_numpy()
    res = {}
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce").fillna(-1e9).to_numpy()
        a = roc_auc_score(y, x)
        res[c] = max(a, 1 - a)
    tab[name] = pd.Series(res)
auc = pd.DataFrame(tab).sort_values("Feb", ascending=False)
auc["in_set"] = ["A" if c in SET_A else "B only" for c in auc.index]
print(auc.round(3).head(12).to_string())
worst = auc[["Feb", "Mar"]].max().max()
assert worst < AUC_TRIPWIRE, f"a single feature reaches AUC {worst:.3f} (>= {AUC_TRIPWIRE}) - almost certainly a leak"
print(f"\nStrongest single feature: AUC {worst:.3f}  (tripwire {AUC_TRIPWIRE})  OK")

# ---------------------------------------------------------------- Feb vs Mar drift (context for out-of-time scoring)
key = ["n_tx", "cancel_rate", "auto_renew_rate", "n_prior_lapses", "days_since_last_tx",
       "days_until_expiry", "last_is_auto_renew", "last_is_cancel", "first_payment_free", "free_share"]
drift = pd.DataFrame({"Feb mean": feb[key].apply(pd.to_numeric, errors="coerce").mean(),
                      "Mar mean": mar[key].apply(pd.to_numeric, errors="coerce").mean()})
print("\nFeature means, Feb vs Mar cohort (a big gap here = out-of-time shift to expect in Phase 7):")
print(drift.round(3).to_string())

null_tab = pd.DataFrame({"Feb % null": 100 * feb[SET_B].isna().mean(), "Mar % null": 100 * mar[SET_B].isna().mean()})
print("\nColumns with >0.5% missing (Feb or Mar):")
print(null_tab[(null_tab > 0.5).any(axis=1)].round(1).to_string())

with open(P("data", "core", "feature_sets.json"), "w") as fh:
    json.dump({"set_a": SET_A, "set_b": SET_B, "categorical": CATEGORICAL, "label": "is_churn"}, fh, indent=2)
print(f"\nWrote {P('data','core','feature_sets.json')}")
print("\nAll Phase 6 (v3) assertions passed.")