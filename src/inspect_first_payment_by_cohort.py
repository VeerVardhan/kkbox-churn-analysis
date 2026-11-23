"""
inspect_first_payment_by_cohort.py  -  why do some cohorts fall off a cliff?  (read-only)

Phase 4 showed:
  * 2015-03: 98.9% retained at k=1 but 31.9% at k=2
  * 2015-06: only 21.2% retained at k=1 (and it is the biggest cohort, 178k)
  * 2016-09/10/11: k=1 retention sliding 46% -> 33% -> 22%
Hypothesis H1: those cohorts are dominated by a different kind of FIRST payment (free ~30-day trial,
a short plan, or a promo) - i.e. the cliff is the "roll-to-pay" step, a known KKBox pattern, not a bug.

For each cohort this prints the make-up of users' first transaction next to the Phase 4 retention.

Reads: data/staging/transactions_v1.parquet, data/core/user_survival.parquet, data/core/cohort_retention.parquet
Writes: nothing
"""
import os
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")

con = duckdb.connect()
con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{P('data','staging','transactions_v1.parquet')}')")
con.execute(f"CREATE VIEW us AS SELECT * FROM read_parquet('{P('data','core','user_survival.parquet')}')")
con.execute(f"CREATE VIEW cr AS SELECT * FROM read_parquet('{P('data','core','cohort_retention.parquet')}')")

# first transaction row per user (staging is unique on (msno, transaction_date))
con.execute("""
    CREATE OR REPLACE TEMP TABLE first_tx AS
    SELECT t.msno, t.payment_plan_days, t.actual_amount_paid, t.is_auto_renew, t.is_cancel,
           u.cohort_month
    FROM tx t JOIN us u ON u.msno = t.msno AND u.first_tx_date = t.transaction_date
    WHERE NOT u.is_left_truncated_cohort
""")
n_first, n_users = con.execute("SELECT COUNT(*), (SELECT COUNT(*) FROM us WHERE NOT is_left_truncated_cohort) FROM first_tx").fetchone()
assert n_first == n_users, f"first-transaction join gave {n_first:,} rows for {n_users:,} users"

print("\n--- First-payment make-up by cohort, next to Phase 4 retention (percent of cohort)")
print(con.execute("""
    WITH f AS (
        SELECT cohort_month, COUNT(*) AS n,
               100.0*AVG((actual_amount_paid = 0)::INT)                    AS pct_free_first,
               100.0*AVG((payment_plan_days <= 7)::INT)                    AS pct_plan_le7d,
               100.0*AVG((payment_plan_days BETWEEN 8 AND 31)::INT)        AS pct_plan_8_31d,
               100.0*AVG((payment_plan_days > 31)::INT)                    AS pct_plan_gt31d,
               100.0*AVG(is_auto_renew::INT)                               AS pct_auto_renew,
               100.0*AVG(is_cancel::INT)                                   AS pct_cancel_flag
        FROM first_tx GROUP BY 1),
    r AS (
        SELECT cohort_month,
               MAX(CASE WHEN k = 1 THEN 100.0*retention_rate END) AS ret_k1,
               MAX(CASE WHEN k = 2 THEN 100.0*retention_rate END) AS ret_k2
        FROM cr GROUP BY 1)
    SELECT strftime(f.cohort_month, '%Y-%m') AS cohort, f.n,
           ROUND(pct_free_first,1) AS free, ROUND(pct_plan_le7d,1) AS le7d, ROUND(pct_plan_8_31d,1) AS d8_31,
           ROUND(pct_plan_gt31d,1) AS gt31d, ROUND(pct_auto_renew,1) AS auto, ROUND(pct_cancel_flag,1) AS cancel,
           ROUND(ret_k1,1) AS ret_k1, ROUND(ret_k2,1) AS ret_k2
    FROM f LEFT JOIN r USING (cohort_month) ORDER BY f.cohort_month
""").df().to_string(index=False))

print("\n--- Within the cliff cohorts: retention by type of first payment (k=1 means survived to day 30, k=2 to day 60)")
print(con.execute("""
    SELECT strftime(f.cohort_month, '%Y-%m') AS cohort,
           CASE WHEN f.actual_amount_paid = 0 THEN 'free first payment'
                WHEN f.payment_plan_days <= 7 THEN 'paid, <=7d plan'
                WHEN f.payment_plan_days <= 31 THEN 'paid, 8-31d plan'
                ELSE 'paid, >31d plan' END AS first_payment,
           COUNT(*) AS users,
           ROUND(100.0*AVG((u.duration_days >= 30)::INT), 1) AS survive_30d,
           ROUND(100.0*AVG((u.duration_days >= 60)::INT), 1) AS survive_60d
    FROM first_tx f JOIN us u USING (msno)
    WHERE strftime(f.cohort_month, '%Y-%m') IN ('2015-03', '2015-06', '2016-10', '2016-11', '2016-01')
    GROUP BY 1, 2 ORDER BY 1, users DESC
""").df().to_string(index=False))