"""
Phase 0 verification gate - KKBox churn dataset.

Four checks, run in order. No pipeline logic here - this script's only job is to
tell you whether it's safe to start building one.

  1. Join coverage      - what % of train_v2 users exist in the other 3 files
  2. Duplicate structure - how many (msno, transaction_date) groups have >1 row,
                           split into "ordinary" duplicates vs ones where
                           membership_expire_date is scattered wildly (likely junk,
                           not a real same-day plan change)
  3. Label reconstruction - re-derive is_churn from transactions_v2 using the
                             official 30-day-no-renewal rule, and compare to the
                             supplied label - but ONLY for users whose 30-day
                             window is fully visible inside transactions_v2's own
                             date range. Users whose last expiry falls too close
                             to the file's cutoff are right-censored from our
                             point of view and are reported separately, not
                             silently counted as disagreements.
  4. Signal check        - a handful of raw features -> 5-fold CV logistic
                           regression ROC-AUC. This is a screen, not final
                           feature engineering: some of these features would not
                           survive the Phase 6 leakage audit. The only question
                           here is "does this dataset carry any signal at all".

NOTE: transaction_date / membership_expire_date / date (logs) arrive as plain
integers in YYYYMMDD form (e.g. 20170331), not DuckDB DATE values. They are
parsed into real DATE columns once, in views, and reused everywhere below.

Run from anywhere - paths are resolved relative to this file's location.
"""

import os
import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")

TRAIN = os.path.join(RAW, "train_v2.csv").replace("\\", "/")
TRANSACTIONS = os.path.join(RAW, "transactions_v2.csv").replace("\\", "/")
LOGS = os.path.join(RAW, "user_logs_v2.csv").replace("\\", "/")
MEMBERS = os.path.join(RAW, "members_v3.csv").replace("\\", "/")

for name, path in [("train_v2", TRAIN), ("transactions_v2", TRANSACTIONS),
                    ("user_logs_v2", LOGS), ("members_v3", MEMBERS)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path} - check data/raw/")

con = duckdb.connect()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Views with properly parsed dates (YYYYMMDD int -> real DATE)
# ---------------------------------------------------------------------------
con.execute(f"""
    CREATE OR REPLACE VIEW tx AS
    SELECT *,
        strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS tx_date,
        strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS expire_date
    FROM read_csv_auto('{TRANSACTIONS}')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW lg AS
    SELECT *,
        strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE AS log_date
    FROM read_csv_auto('{LOGS}')
""")

MAX_DATE = con.execute("SELECT MAX(tx_date) AS d FROM tx").fetchdf()["d"][0]
print(f"transactions_v2 max transaction date (parsed): {MAX_DATE}")

# ---------------------------------------------------------------------------
# CHECK 1 - join coverage
# ---------------------------------------------------------------------------
section("CHECK 1: join coverage of train_v2 users across the other 3 files")

coverage = con.execute(f"""
    WITH t  AS (SELECT DISTINCT msno FROM read_csv_auto('{TRAIN}')),
         txd AS (SELECT DISTINCT msno FROM tx),
         mb AS (SELECT DISTINCT msno FROM read_csv_auto('{MEMBERS}')),
         lgd AS (SELECT DISTINCT msno FROM lg)
    SELECT
        COUNT(*)         AS train_users,
        COUNT(txd.msno)  AS in_transactions,
        COUNT(mb.msno)   AS in_members,
        COUNT(lgd.msno)  AS in_logs
    FROM t
    LEFT JOIN txd ON txd.msno = t.msno
    LEFT JOIN mb  ON mb.msno = t.msno
    LEFT JOIN lgd ON lgd.msno = t.msno
""").fetchdf()
print(coverage.to_string(index=False))
n = coverage["train_users"][0]
for col in ["in_transactions", "in_members", "in_logs"]:
    print(f"  {col}: {100 * coverage[col][0] / n:.1f}% coverage")

train_churn_rate = con.execute(f"""
    SELECT AVG(is_churn) AS churn_rate, COUNT(*) AS n FROM read_csv_auto('{TRAIN}')
""").fetchdf()
print(f"\n  train_v2 churn rate: {train_churn_rate['churn_rate'][0]*100:.2f}% "
      f"({train_churn_rate['n'][0]} users)")

# ---------------------------------------------------------------------------
# CHECK 2 - duplicate (msno, transaction_date) structure
# ---------------------------------------------------------------------------
section("CHECK 2: duplicate transaction rows (same msno + transaction_date)")

dup_summary = con.execute("""
    WITH dup AS (
        SELECT msno, transaction_date, COUNT(*) AS n,
               MIN(expire_date) AS min_exp, MAX(expire_date) AS max_exp
        FROM tx
        GROUP BY msno, transaction_date
    )
    SELECT
        COUNT(*) AS total_groups,
        SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) AS groups_with_dupes,
        SUM(CASE WHEN n > 1 THEN n ELSE 0 END) AS rows_in_dupe_groups,
        MAX(n) AS max_rows_in_one_group,
        SUM(CASE WHEN n > 1 AND DATE_DIFF('day', min_exp, max_exp) > 365
                 THEN 1 ELSE 0 END) AS groups_with_wild_expiry_spread
    FROM dup
""").fetchdf()
print(dup_summary.to_string(index=False))
pct_dup_groups = 100 * dup_summary["groups_with_dupes"][0] / dup_summary["total_groups"][0]
pct_wild = 100 * dup_summary["groups_with_wild_expiry_spread"][0] / max(dup_summary["groups_with_dupes"][0], 1)
print(f"  {pct_dup_groups:.2f}% of (msno, date) groups have more than one row")
print(f"  Of those duplicate groups, {pct_wild:.1f}% have membership_expire_date "
      f"values spread over more than a year - likely junk rows, not legitimate "
      f"same-day plan changes. Inspect a sample below.")

print("\nSample ORDINARY duplicate group (expiry dates close together):")
ordinary = con.execute("""
    WITH dup AS (
        SELECT msno, transaction_date, COUNT(*) AS n,
               DATE_DIFF('day', MIN(expire_date), MAX(expire_date)) AS spread_days
        FROM tx
        GROUP BY msno, transaction_date
        HAVING COUNT(*) > 1 AND DATE_DIFF('day', MIN(expire_date), MAX(expire_date)) <= 365
    )
    SELECT msno, transaction_date FROM dup ORDER BY n DESC LIMIT 1
""").fetchdf()
if len(ordinary):
    row = ordinary.iloc[0]
    sample = con.execute(
        "SELECT msno, payment_method_id, payment_plan_days, plan_list_price, "
        "actual_amount_paid, is_auto_renew, transaction_date, membership_expire_date, is_cancel "
        "FROM tx WHERE msno = ? AND transaction_date = ? ORDER BY membership_expire_date",
        [str(row["msno"]), int(row["transaction_date"])]
    ).fetchdf()
    print(sample.to_string(index=False))
else:
    print("  (none found - all duplicate groups have wide expiry spread)")

print("\nSample WILD-SPREAD duplicate group (likely junk, not a real plan change):")
wild = con.execute("""
    WITH dup AS (
        SELECT msno, transaction_date, COUNT(*) AS n,
               DATE_DIFF('day', MIN(expire_date), MAX(expire_date)) AS spread_days
        FROM tx
        GROUP BY msno, transaction_date
        HAVING COUNT(*) > 1 AND DATE_DIFF('day', MIN(expire_date), MAX(expire_date)) > 365
    )
    SELECT msno, transaction_date, n, spread_days FROM dup ORDER BY n DESC LIMIT 1
""").fetchdf()
if len(wild):
    row = wild.iloc[0]
    print(f"  msno={row['msno']}  n_rows={row['n']}  expiry_spread_days={row['spread_days']}")
    sample = con.execute(
        "SELECT payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, "
        "is_auto_renew, is_cancel, membership_expire_date "
        "FROM tx WHERE msno = ? AND transaction_date = ? ORDER BY membership_expire_date LIMIT 10",
        [str(row["msno"]), int(row["transaction_date"])]
    ).fetchdf()
    print(sample.to_string(index=False))
    print("  (showing first 10 of", int(row["n"]), "rows)")
else:
    print("  (none found)")

# ---------------------------------------------------------------------------
# CHECK 3 - label reconstruction (censoring-aware)
# ---------------------------------------------------------------------------
section("CHECK 3: re-derive is_churn from transactions_v2, compare to train_v2")

recon = con.execute(f"""
    WITH last_expire AS (
        SELECT msno, expire_date AS last_expire_date
        FROM tx
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY msno ORDER BY tx_date DESC, expire_date DESC
        ) = 1
    ),
    visible AS (
        -- only users whose 30-day renewal window is fully inside the file's own date range
        SELECT msno, last_expire_date
        FROM last_expire
        WHERE last_expire_date + INTERVAL 30 DAY <= DATE '{MAX_DATE}'
    ),
    renewed AS (
        SELECT DISTINCT v.msno
        FROM visible v
        JOIN tx t
          ON t.msno = v.msno
         AND t.tx_date > v.last_expire_date
         AND t.tx_date <= v.last_expire_date + INTERVAL 30 DAY
    ),
    reconstructed AS (
        SELECT v.msno,
               CASE WHEN r.msno IS NULL THEN 1 ELSE 0 END AS reconstructed_churn
        FROM visible v
        LEFT JOIN renewed r ON r.msno = v.msno
    )
    SELECT tr.is_churn AS supplied_churn, rc.reconstructed_churn, COUNT(*) AS n
    FROM reconstructed rc
    JOIN read_csv_auto('{TRAIN}') tr ON tr.msno = rc.msno
    GROUP BY 1, 2
    ORDER BY 1, 2
""").fetchdf()
print(recon.to_string(index=False))

total_checked = recon["n"].sum()
agree = recon[recon["supplied_churn"] == recon["reconstructed_churn"]]["n"].sum()
print(f"\n  Agreement on the visible (non-censored) subset: "
      f"{agree}/{total_checked} = {100*agree/total_checked:.1f}%")

visible_count = con.execute(f"""
    WITH last_expire AS (
        SELECT msno, expire_date AS last_expire_date
        FROM tx
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY msno ORDER BY tx_date DESC, expire_date DESC
        ) = 1
    )
    SELECT
        COUNT(*) AS total_users_with_transactions,
        SUM(CASE WHEN last_expire_date + INTERVAL 30 DAY <= DATE '{MAX_DATE}'
                 THEN 1 ELSE 0 END) AS visible_for_reconstruction
    FROM last_expire
""").fetchdf()
print(f"  Users with transactions: {visible_count['total_users_with_transactions'][0]}, "
      f"of which {visible_count['visible_for_reconstruction'][0]} have a fully-visible "
      f"30-day window (the rest are right-censored - cannot be checked from this file alone, "
      f"this is expected, not a bug).")

# ---------------------------------------------------------------------------
# CHECK 4 - signal check (raw screen, not final features)
# ---------------------------------------------------------------------------
section("CHECK 4: signal check - raw features -> 5-fold CV logistic regression AUC")

features = con.execute(f"""
    WITH tx_agg AS (
        SELECT
            msno,
            COUNT(*)                                     AS n_transactions,
            AVG(payment_plan_days)                        AS avg_plan_days,
            AVG(actual_amount_paid)                        AS avg_amount_paid,
            AVG(plan_list_price - actual_amount_paid)      AS avg_discount,
            AVG(CAST(is_cancel AS DOUBLE))                 AS cancel_rate,
            AVG(CAST(is_auto_renew AS DOUBLE))             AS auto_renew_rate,
            DATE_DIFF('day', MAX(tx_date), DATE '{MAX_DATE}') AS days_since_last_tx
        FROM tx
        GROUP BY msno
    ),
    log_agg AS (
        SELECT
            msno,
            COUNT(*)                                       AS n_active_days,
            AVG(total_secs)                                 AS avg_total_secs,
            AVG(num_unq)                                    AS avg_num_unq,
            AVG(CAST(num_100 AS DOUBLE) /
                NULLIF(num_25+num_50+num_75+num_985+num_100, 0)) AS avg_completion_ratio
        FROM lg
        GROUP BY msno
    )
    SELECT
        tr.msno, tr.is_churn,
        tx.n_transactions, tx.avg_plan_days, tx.avg_amount_paid, tx.avg_discount,
        tx.cancel_rate, tx.auto_renew_rate, tx.days_since_last_tx,
        lg.n_active_days, lg.avg_total_secs, lg.avg_num_unq, lg.avg_completion_ratio,
        mb.city, mb.bd, mb.gender, mb.registered_via
    FROM read_csv_auto('{TRAIN}') tr
    LEFT JOIN tx_agg tx ON tx.msno = tr.msno
    LEFT JOIN log_agg lg ON lg.msno = tr.msno
    LEFT JOIN read_csv_auto('{MEMBERS}') mb ON mb.msno = tr.msno
""").fetchdf()

print(f"  Feature table: {features.shape[0]} rows, {features.shape[1]} columns")
print(f"  Missing values per column:\n{features.isna().sum().to_string()}")

y = features["is_churn"]
num_cols = ["n_transactions", "avg_plan_days", "avg_amount_paid", "avg_discount",
            "cancel_rate", "auto_renew_rate", "days_since_last_tx",
            "n_active_days", "avg_total_secs", "avg_num_unq", "avg_completion_ratio", "bd"]
X = features[num_cols].copy()

gender_dummies = pd.get_dummies(features["gender"], prefix="gender", dummy_na=True)
X = pd.concat([X, gender_dummies], axis=1)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(f"\n  5-fold CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"  Individual folds: {np.round(scores, 3)}")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
section("GATE VERDICT")
agree_pct = 100 * agree / total_checked
print(f"  Join coverage (transactions/members/logs): see Check 1 above")
print(f"  Duplicate groups: {pct_dup_groups:.2f}% of all groups "
      f"({pct_wild:.1f}% of those look like junk, not real same-day changes)")
print(f"  Label reconstruction agreement (visible subset): {agree_pct:.1f}%  "
      f"(expect ~90%+, per published replications)")
print(f"  Signal check CV AUC: {scores.mean():.3f} +/- {scores.std():.3f}  "
      f"(need clearly above 0.60 to proceed; near 0.5 means stop)")
print("\n  Read all four numbers above before deciding whether to proceed to Phase 1 staging.")