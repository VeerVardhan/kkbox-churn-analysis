"""
check_phase3_revenue.py  -  Phase 3 revenue reconciliation (read-only)

Why this exists: build_fct_member_month.py's own "ratio" check divided the cohort's
revenue by the payments of ALL users in the transaction file, so it was meaningless.
This script reconciles properly, for cohort users only:

  raw payments  ->  minus tail beyond as-of  ->  minus months before spine start  ->  fct total

Expected: fct total == "expected (clipped)" to within rounding (ratio 1.000).
The other two ratios tell you how much revenue is genuinely outside the window.

Reads : data/core/dim_member_{feb,mar}.parquet, data/core/fct_member_month_{feb,mar}.parquet(.parquet),
        data/staging/transactions_v1.parquet, data/staging/transactions.parquet
"""
import os
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")


def pick(name):
    """fct file may still carry the accidental double extension."""
    good, bad = P("data", "core", name + ".parquet"), P("data", "core", name + ".parquet.parquet")
    return good if os.path.exists(good) else bad


con = duckdb.connect()
con.execute(f"CREATE VIEW tx_v1 AS SELECT * FROM read_parquet('{P('data','staging','transactions_v1.parquet')}')")
con.execute(f"CREATE VIEW tx_v2 AS SELECT * FROM read_parquet('{P('data','staging','transactions.parquet')}')")

COHORTS = [
    ("Feb", "dim_member_feb", "fct_member_month_feb", "2017-01-31",
     "SELECT msno, transaction_date, payment_plan_days, actual_amount_paid FROM tx_v1 "
     "WHERE transaction_date <= DATE '2017-01-31'"),
    ("Mar", "dim_member_mar", "fct_member_month_mar", "2017-02-28",
     "SELECT msno, transaction_date, payment_plan_days, actual_amount_paid FROM tx_v1 "
     "WHERE transaction_date <= DATE '2017-02-28' UNION ALL "
     "SELECT msno, transaction_date, payment_plan_days, actual_amount_paid FROM tx_v2 "
     "WHERE transaction_date <= DATE '2017-02-28'"),
]

for label, dim, fct, as_of, tx_sql in COHORTS:
    print("\n" + "=" * 70 + f"\n{label} cohort (as-of {as_of})\n" + "=" * 70)
    con.execute(f"CREATE OR REPLACE VIEW d AS SELECT msno, registration_date FROM read_parquet('{P('data','core',dim+'.parquet')}')")
    con.execute(f"CREATE OR REPLACE VIEW f AS SELECT * FROM read_parquet('{pick(fct)}')")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE t AS
        SELECT x.msno, x.transaction_date AS s,
               x.transaction_date + CAST(x.payment_plan_days AS INT) AS e,
               x.actual_amount_paid::DOUBLE AS paid,
               x.actual_amount_paid::DOUBLE / x.payment_plan_days AS rate,
               CASE WHEN x.payment_plan_days <= 7 THEN 'weekly_or_shorter'
                    WHEN x.payment_plan_days <= 31 THEN 'monthly'
                    WHEN x.payment_plan_days <= 100 THEN 'quarterly'
                    WHEN x.payment_plan_days <= 200 THEN 'semi_annual'
                    ELSE 'annual_or_longer' END AS tier,
               GREATEST(COALESCE(CAST(DATE_TRUNC('month', d.registration_date) AS DATE), DATE '2015-01-01'),
                        DATE '2015-01-01') AS spine_start
        FROM ({tx_sql}) x JOIN d ON d.msno = x.msno
        WHERE x.payment_plan_days > 0
    """)
    cap = f"(DATE '{as_of}' + 1)"
    r = con.execute(f"""
        SELECT SUM(paid),
               SUM(rate * GREATEST(0, DATE_DIFF('day', s, LEAST(e, {cap})))),
               SUM(rate * GREATEST(0, DATE_DIFF('day', GREATEST(s, spine_start), LEAST(e, {cap}))))
        FROM t""").fetchone()
    raw, unclipped, clipped = r
    fct_total = con.execute("SELECT SUM(normalized_revenue) FROM f").fetchone()[0]
    print(f"raw payments (cohort users only)        : {raw:>16,.0f}")
    print(f"expected, pre-as-of share only          : {unclipped:>16,.0f}   ({unclipped/raw:.3f} of raw; rest = tail beyond as-of)")
    print(f"expected, also dropping pre-spine months: {clipped:>16,.0f}   ({clipped/unclipped:.4f} of previous)")
    print(f"fct_member_month total                  : {fct_total:>16,.0f}   (fct / expected = {fct_total/clipped:.5f})")
    print("\nby plan tier: share of raw payments that is pre-as-of")
    print(con.execute(f"""
        SELECT tier, ROUND(SUM(paid)) AS raw,
               ROUND(SUM(rate*GREATEST(0, DATE_DIFF('day', s, LEAST(e, {cap})))) / SUM(paid), 3) AS pre_asof_share
        FROM t GROUP BY 1 ORDER BY raw DESC""").df().to_string(index=False))
    ok = abs(fct_total / clipped - 1) < 0.001
    print(f"\nreconciliation {'PASSED' if ok else 'FAILED - do not proceed'} (tolerance 0.1%)")