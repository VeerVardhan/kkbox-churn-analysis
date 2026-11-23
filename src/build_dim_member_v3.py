"""
src/build_dim_member_v3.py  -  Phase 2 (v3): one row per user, for BOTH cohorts.

Descriptive dimension only - not a feature table. Every "as of X" column is computed
strictly from transactions dated on or before that cohort's as-of date (Decision 6, v3):
  - Feb cohort: as-of 2017-01-31, transactions.csv only.
  - Mar cohort: as-of 2017-02-28, transactions.csv + pre-cutoff rows of transactions_v2.csv.

The re-derived churn flag is separate from the as-of features: it uses the FULL available
history in each cohort's own label file (because reconstructing the label requires seeing
transactions after the expiry, which the features must never see). This mirrors Phase 0
check 3 exactly, so both cohorts self-assert against their own check-3 numbers.

Reads : data/staging/{train_feb,transactions_v1,train,transactions,members}.parquet
Writes: data/core/dim_member_feb.parquet, data/core/dim_member_mar.parquet
"""
import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(PROJECT_ROOT, "data", "staging")
CORE = os.path.join(PROJECT_ROOT, "data", "core")
os.makedirs(CORE, exist_ok=True)

AGE_MIN, AGE_MAX = 10, 100

# Phase 0 (v3) check-3 numbers, to self-assert against. Update if check 3 is re-run.
FEB_VISIBLE, FEB_AGREE = 7017, 6867
MAR_VISIBLE, MAR_AGREE = 377, 347   # the original v2 Phase 0 numbers


def pq(name):
    return os.path.join(STAGING, name + ".parquet").replace("\\", "/")


con = duckdb.connect()
con.execute(f"CREATE VIEW members AS SELECT * FROM read_parquet('{pq('members')}')")


def build_cohort(label, train_tbl, as_of, feature_tx_sql, label_tx_sql,
                  out_name, expect_visible, expect_agree):
    print("\n" + "=" * 78)
    print(f"COHORT: {label}  (as-of {as_of})")
    print("=" * 78)

    con.execute(f"CREATE OR REPLACE VIEW cohort_train AS SELECT * FROM {train_tbl}")
    con.execute(f"CREATE OR REPLACE VIEW cohort_tx_features AS {feature_tx_sql}")
    con.execute(f"CREATE OR REPLACE VIEW cohort_tx_label AS {label_tx_sql}")

    # transaction summary, built ONLY from rows on/before the as-of date
    con.execute("""
        CREATE OR REPLACE TEMP TABLE tx_last_asof AS
        SELECT msno, transaction_date AS last_tx_date, membership_expire_date AS expiry_asof
        FROM cohort_tx_features
        QUALIFY ROW_NUMBER() OVER (PARTITION BY msno
                                   ORDER BY transaction_date DESC, membership_expire_date DESC) = 1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE tx_agg_asof AS
        SELECT msno, COUNT(*) AS n_transactions, MIN(transaction_date) AS first_tx_date
        FROM cohort_tx_features GROUP BY msno
    """)

    # label reconstruction uses the FULL label file (post-as-of transactions allowed -
    # this is the outcome, never a feature)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE tx_last_full AS
        SELECT msno, transaction_date AS last_tx_date_full, membership_expire_date AS last_expiry_full
        FROM cohort_tx_label
        QUALIFY ROW_NUMBER() OVER (PARTITION BY msno
                                   ORDER BY transaction_date DESC, membership_expire_date DESC) = 1
    """)
    max_label_date = con.execute("SELECT MAX(transaction_date) FROM cohort_tx_label").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE renewed AS
        SELECT DISTINCT l.msno
        FROM tx_last_full l
        JOIN cohort_tx_label t
          ON t.msno = l.msno
         AND t.transaction_date >  l.last_expiry_full
         AND t.transaction_date <= l.last_expiry_full + INTERVAL 30 DAY
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dim AS
        SELECT
            tr.msno,
            tr.is_churn                                              AS is_churn_supplied,
            (m.msno IS NOT NULL)                                     AS has_member_record,
            m.registration_date,
            CAST(DATE_TRUNC('month', m.registration_date) AS DATE)   AS registration_month,
            m.city, m.registered_via,
            m.bd                                                     AS bd_raw,
            CASE WHEN m.msno IS NULL THEN 'no_member_record'
                 WHEN m.bd IS NULL   THEN 'missing'
                 WHEN m.bd <= 0      THEN 'zero_or_negative'
                 WHEN m.bd BETWEEN {AGE_MIN} AND {AGE_MAX} THEN 'plausible'
                 ELSE 'implausible' END                              AS age_flag,
            CASE WHEN m.bd BETWEEN {AGE_MIN} AND {AGE_MAX} THEN m.bd END AS age_clean,
            CASE WHEN m.msno IS NULL   THEN 'no_member_record'
                 WHEN m.gender IS NULL THEN 'not_provided'
                 ELSE m.gender END                                   AS gender_clean,

            (a.msno IS NOT NULL)                                     AS has_pre_asof_transactions,
            COALESCE(a.n_transactions, 0)                            AS n_transactions_asof,
            a.first_tx_date                                          AS first_tx_date_asof,
            la.last_tx_date                                          AS last_tx_date_asof,
            la.expiry_asof,

            COALESCE(m.registration_date > DATE '{as_of}', FALSE)    AS registration_after_asof,
            -- NULL (not negative) when registration falls after the as-of date: the cohort is
            -- defined by expiry month, not registration month, so a user who signs up mid-window
            -- can still appear here. "Tenure before you existed" is not a number, it's a flag -
            -- see registration_after_asof above.
            CASE WHEN m.registration_date IS NOT NULL AND m.registration_date <= DATE '{as_of}'
                 THEN DATE_DIFF('day', m.registration_date, DATE '{as_of}') END AS tenure_days_asof,

            lf.last_expiry_full,
            COALESCE((lf.last_expiry_full + INTERVAL 30 DAY) <= DATE '{max_label_date}', FALSE) AS window_visible
        FROM cohort_train tr
        LEFT JOIN members      m  ON m.msno = tr.msno
        LEFT JOIN tx_agg_asof  a  ON a.msno = tr.msno
        LEFT JOIN tx_last_asof la ON la.msno = tr.msno
        LEFT JOIN tx_last_full lf ON lf.msno = tr.msno
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE dim_final AS
        SELECT d.*,
               NOT d.window_visible AS is_window_censored,
               CASE WHEN d.window_visible
                    THEN CASE WHEN r.msno IS NULL THEN 1 ELSE 0 END END AS reconstructed_churn
        FROM dim d
        LEFT JOIN renewed r ON r.msno = d.msno
    """)

    out_path = os.path.join(CORE, out_name).replace("\\", "/")
    con.execute(f"COPY dim_final TO '{out_path}' (FORMAT PARQUET)")
    print(f"Wrote {out_path}")

    # --- assertions ---
    n_rows, n_users = con.execute("SELECT COUNT(*), COUNT(DISTINCT msno) FROM dim_final").fetchone()
    n_train = con.execute("SELECT COUNT(*) FROM cohort_train").fetchone()[0]
    assert n_rows == n_users == n_train, f"not 1:1 with cohort: {n_rows}, {n_users}, {n_train}"

    neg_tenure = con.execute(
        "SELECT COUNT(*) FROM dim_final WHERE tenure_days_asof < 0").fetchone()[0]
    assert neg_tenure == 0, f"{neg_tenure} users with negative tenure as of {as_of} (should be impossible - nulled instead)"

    null_tenure_after_reg = con.execute("""
        SELECT COUNT(*) FROM dim_final
        WHERE tenure_days_asof IS NULL AND registration_after_asof
    """).fetchone()[0]
    n_reg_after = con.execute(
        "SELECT COUNT(*) FROM dim_final WHERE registration_after_asof").fetchone()[0]
    assert null_tenure_after_reg == n_reg_after, \
        "tenure_days_asof should be NULL for every registration_after_asof user, and only those"
    print(f"{n_reg_after} users registered after the as-of date (new signups landing in this "
          f"cohort's expiry window) - tenure_days_asof is NULL for these, by design.")

    bad_visible = con.execute(f"""
        SELECT COUNT(*) FROM dim_final
        WHERE window_visible AND (last_expiry_full + INTERVAL 30 DAY) > DATE '{max_label_date}'
    """).fetchone()[0]
    assert bad_visible == 0, "window_visible inconsistent with the label file's own cutoff"

    bad_recon = con.execute("""
        SELECT COUNT(*) FROM dim_final
        WHERE (window_visible AND reconstructed_churn IS NULL)
           OR (NOT window_visible AND reconstructed_churn IS NOT NULL)
    """).fetchone()[0]
    assert bad_recon == 0, "reconstructed_churn must exist exactly where window_visible"

    visible, agree = con.execute("""
        SELECT COUNT(*), SUM((reconstructed_churn = is_churn_supplied)::INT)
        FROM dim_final WHERE window_visible
    """).fetchone()
    agree = int(agree)
    print(f"Label agreement on visible subset: {agree}/{visible} = {100*agree/visible:.1f}% "
          f"(Phase 0 check 3: {expect_agree}/{expect_visible})")
    assert (visible, agree) == (expect_visible, expect_agree), \
        f"Visible/agree counts differ from Phase 0 check 3 for {label} - the rule was not " \
        f"reproduced exactly; investigate before continuing"

    def show(title, sql):
        print(f"\n--- {title}")
        print(con.execute(sql).df().to_string(index=False))

    show("age_flag", "SELECT age_flag, COUNT(*) AS users FROM dim_final GROUP BY 1 ORDER BY users DESC")
    show("gender_clean", "SELECT gender_clean, COUNT(*) AS users FROM dim_final GROUP BY 1 ORDER BY users DESC")
    show("has_pre_asof_transactions", """
        SELECT has_pre_asof_transactions, COUNT(*) AS users,
               ROUND(100.0*AVG(is_churn_supplied),1) AS churn_pct
        FROM dim_final GROUP BY 1""")
    show("tenure_days_asof", """
        SELECT MIN(tenure_days_asof) AS min_t, QUANTILE_CONT(tenure_days_asof,0.5) AS median_t,
               MAX(tenure_days_asof) AS max_t
        FROM dim_final""")
    show("registration_after_asof (should be rare/zero)",
         "SELECT registration_after_asof, COUNT(*) AS users FROM dim_final GROUP BY 1")

    print(f"\nAll Phase 2 (v3) assertions passed for {label}.")


# ---------------------------------------------------------------------------
# Feb cohort: as-of 2017-01-31, transactions.csv (v1) only
# ---------------------------------------------------------------------------
con.execute(f"CREATE OR REPLACE VIEW tx_v1 AS SELECT * FROM read_parquet('{pq('transactions_v1')}')")
con.execute(f"CREATE OR REPLACE VIEW train_feb AS SELECT * FROM read_parquet('{pq('train_feb')}')")

build_cohort(
    label="Feb (train.csv)",
    train_tbl="train_feb",
    as_of="2017-01-31",
    feature_tx_sql="SELECT * FROM tx_v1 WHERE transaction_date <= DATE '2017-01-31'",
    label_tx_sql="SELECT * FROM tx_v1",   # full v1 file, ends 2017-02-28
    out_name="dim_member_feb.parquet",
    expect_visible=FEB_VISIBLE, expect_agree=FEB_AGREE,
)

# ---------------------------------------------------------------------------
# Mar cohort: as-of 2017-02-28, transactions.csv + pre-cutoff rows of transactions_v2.csv
# ---------------------------------------------------------------------------
con.execute(f"CREATE OR REPLACE VIEW tx_v2 AS SELECT * FROM read_parquet('{pq('transactions')}')")
con.execute(f"CREATE OR REPLACE VIEW train_mar AS SELECT * FROM read_parquet('{pq('train')}')")

build_cohort(
    label="Mar (train_v2.csv)",
    train_tbl="train_mar",
    as_of="2017-02-28",
    feature_tx_sql="""
        SELECT msno, transaction_date, membership_expire_date FROM tx_v1
        WHERE transaction_date <= DATE '2017-02-28'
        UNION ALL
        SELECT msno, transaction_date, membership_expire_date FROM tx_v2
        WHERE transaction_date <= DATE '2017-02-28'
    """,
    label_tx_sql="SELECT * FROM tx_v2",   # full v2 file, ends 2017-03-31 - needed to see the renewal
    out_name="dim_member_mar.parquet",
    expect_visible=MAR_VISIBLE, expect_agree=MAR_AGREE,
)

print("\nBoth cohorts built. Phase 2 (v3) gate: OK to proceed to Phase 3.")