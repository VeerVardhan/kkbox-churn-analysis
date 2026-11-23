"""
src/build_fct_member_month.py  -  Phase 3 (v3): month-by-month state per user, both cohorts.

One row per (user, month), from max(registration month, 2015-01) to the cohort's as-of month.
All input periods come from the same pre-as-of transaction views used in Phase 2, so this
table carries no information past the as-of date - it is still a descriptive table, not a
feature table (Phase 6 decides what from here is safe to model with).

Revenue policy (Decision 4 / Phase 3 decide): prepaid revenue is SPREAD evenly across the
days of its plan, not booked at transaction date. A transaction for payment_plan_days D at
actual_amount_paid A contributes A/D per day to every calendar day its period covers; a
user-month's normalized_revenue is the sum of that daily rate across the days of the period
that fall in that month. This is normalized MRR (Decision 4) rolled up to monthly granularity,
not a sum of payments - a 410-day prepayment adds ~1/13.7 of its value to each month it spans.

"Governing period" for plan tier / auto-renew / discount in a month = the period whose
transaction_date is the latest one on or before that month's end - i.e. the plan state as it
stood at the close of that month. cancel_occurred is transaction-level, not period-level: it
fires in whichever month the cancelling transaction itself is dated.

Reads : data/core/dim_member_{feb,mar}.parquet, data/staging/{transactions_v1,transactions}.parquet
Writes: data/core/fct_member_month_feb.parquet, data/core/fct_member_month_mar.parquet
"""
import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(PROJECT_ROOT, "data", "staging")
CORE = os.path.join(PROJECT_ROOT, "data", "core")

HISTORY_START = "2015-01-01"   # transactions.csv's own min date - nothing exists before this


def pq_staging(name):
    return os.path.join(STAGING, name + ".parquet").replace("\\", "/")


def pq_core(name):
    return os.path.join(CORE, name + ".parquet").replace("\\", "/")


con = duckdb.connect()
con.execute(f"CREATE VIEW tx_v1 AS SELECT * FROM read_parquet('{pq_staging('transactions_v1')}')")
con.execute(f"CREATE VIEW tx_v2 AS SELECT * FROM read_parquet('{pq_staging('transactions')}')")


def build_cohort(label, dim_name, as_of, feature_tx_sql, out_name):
    print("\n" + "=" * 78)
    print(f"COHORT: {label}  (as-of {as_of})")
    print("=" * 78)

    con.execute(f"CREATE OR REPLACE VIEW cohort_dim AS SELECT * FROM read_parquet('{pq_core(dim_name)}')")
    con.execute(f"CREATE OR REPLACE VIEW periods AS {feature_tx_sql}")

    # one row per transaction = one subscription period, with a normalized daily rate
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE period_rates AS
        SELECT msno, transaction_date, membership_expire_date,
               payment_plan_days, plan_list_price, actual_amount_paid,
               is_auto_renew, is_cancel,
               transaction_date AS period_start,
               transaction_date + CAST(payment_plan_days AS INT) AS period_end_excl,
               CASE WHEN payment_plan_days > 0
                    THEN actual_amount_paid::DOUBLE / payment_plan_days END AS daily_rate,
               CASE WHEN payment_plan_days <= 7   THEN 'weekly_or_shorter'
                    WHEN payment_plan_days <= 31  THEN 'monthly'
                    WHEN payment_plan_days <= 100 THEN 'quarterly'
                    WHEN payment_plan_days <= 200 THEN 'semi_annual'
                    ELSE 'annual_or_longer' END AS plan_tier
        FROM periods
        WHERE payment_plan_days > 0   -- a handful of zero-length plans can't be spread; excluded, not silently kept
    """)

    # month spine per user: max(registration month, 2015-01) to the cohort's as-of month
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE spine AS
        SELECT d.msno,
               GREATEST(COALESCE(DATE_TRUNC('month', d.registration_date), DATE '{HISTORY_START}'),
                        DATE '{HISTORY_START}') AS spine_start,
               DATE_TRUNC('month', DATE '{as_of}') AS spine_end
        FROM cohort_dim d
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE user_months AS
        SELECT msno, UNNEST(generate_series(spine_start, spine_end, INTERVAL 1 MONTH)) AS month_start
        FROM spine
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE user_months_bounded AS
        SELECT msno, month_start,
               month_start + INTERVAL 1 MONTH AS month_end_excl,
               CAST(month_start + INTERVAL 1 MONTH - INTERVAL 1 DAY AS DATE) AS month_last_day
        FROM user_months
    """)

    # revenue: split each period into the calendar months it touches (~1.3 rows per period),
    # compute overlap days per (period, month), then sum per (user, month). This replaces a
    # user-month x period range join that produced hundreds of millions of candidate pairs.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE period_months AS
        SELECT msno, daily_rate, period_start, period_end_excl,
               UNNEST(generate_series(DATE_TRUNC('month', period_start),
                                      DATE_TRUNC('month', period_end_excl - 1),
                                      INTERVAL 1 MONTH)) AS month_start
        FROM period_rates
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE revenue_by_month AS
        SELECT msno, CAST(month_start AS DATE) AS month_start,
               SUM(GREATEST(0, DATE_DIFF('day',
                       GREATEST(period_start, CAST(month_start AS DATE)),
                       LEAST(period_end_excl, CAST(month_start + INTERVAL 1 MONTH AS DATE))))
                   * daily_rate) AS normalized_revenue,
               COUNT(*) AS n_overlapping_periods
        FROM period_months
        GROUP BY msno, CAST(month_start AS DATE)
    """)

    # governing period: the plan state as known at the close of each month. ASOF JOIN picks,
    # per user-month, the single latest transaction dated on or before the month's last day.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE governing AS
        SELECT um.msno, um.month_start,
               p.plan_tier, p.is_auto_renew,
               (p.plan_list_price - p.actual_amount_paid) AS discount_depth,
               (p.period_end_excl > um.month_end_excl) AS membership_in_force_eom
        FROM user_months_bounded um
        ASOF JOIN period_rates p
          ON um.msno = p.msno AND um.month_last_day >= p.transaction_date
    """)

    # cancel: transaction-level event, keyed to the month the cancelling transaction is dated in
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cancels AS
        SELECT msno, DATE_TRUNC('month', transaction_date) AS month_start, 1 AS cancel_occurred
        FROM period_rates
        WHERE is_cancel = 1
        GROUP BY 1, 2
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fct AS
        SELECT
            um.msno, um.month_start,
            COALESCE(rv.normalized_revenue, 0) AS normalized_revenue,
            COALESCE(rv.n_overlapping_periods, 0) AS n_overlapping_periods,
            COALESCE(g.membership_in_force_eom, FALSE) AS membership_in_force_eom,
            g.plan_tier, g.is_auto_renew, g.discount_depth,
            COALESCE(c.cancel_occurred, 0) AS cancel_occurred
        FROM user_months_bounded um
        LEFT JOIN revenue_by_month rv ON rv.msno = um.msno AND rv.month_start = um.month_start
        LEFT JOIN governing        g  ON g.msno  = um.msno AND g.month_start  = um.month_start
        LEFT JOIN cancels          c  ON c.msno  = um.msno AND c.month_start  = um.month_start
    """)

    out_path = pq_core(out_name)
    con.execute(f"COPY fct TO '{out_path}' (FORMAT PARQUET)")
    print(f"Wrote {out_path}")

    # --- assertions (Phase 3 gate) ---
    dup = con.execute("""
        SELECT COUNT(*) FROM (SELECT msno, month_start, COUNT(*) AS n
                               FROM fct GROUP BY 1, 2 HAVING COUNT(*) > 1)
    """).fetchone()[0]
    assert dup == 0, f"{dup} duplicate (msno, month) rows"

    neg_rev = con.execute("SELECT COUNT(*) FROM fct WHERE normalized_revenue < 0").fetchone()[0]
    assert neg_rev == 0, f"{neg_rev} user-months with negative normalized revenue"

    overlap_flag = con.execute(
        "SELECT COUNT(*) FROM fct WHERE n_overlapping_periods > 1").fetchone()[0]
    print(f"{overlap_flag:,} user-months had more than one overlapping transaction period "
          f"(mid-month plan changes or renewals - summed, not an error, but worth sampling "
          f"if this is a large share).")

    n_months, n_users = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT msno) FROM fct").fetchone()
    total_norm_rev = con.execute("SELECT SUM(normalized_revenue) FROM fct").fetchone()[0]
    total_paid = con.execute(
        "SELECT SUM(actual_amount_paid) FROM period_rates WHERE msno IN (SELECT msno FROM cohort_dim)").fetchone()[0]
    print(f"{n_months:,} user-months across {n_users:,} users.")
    print(f"Total normalized revenue: {total_norm_rev:,.0f}  |  Total raw payments: {total_paid:,.0f}  "
          f"(ratio {total_norm_rev/total_paid:.3f} - expect close to 1.0; drift comes from periods "
          f"whose tail extends past the as-of date, since only the pre-as-of share is counted here)")

    def show(title, sql):
        print(f"\n--- {title}")
        print(con.execute(sql).df().to_string(index=False))

    show("plan_tier", "SELECT plan_tier, COUNT(*) AS user_months FROM fct GROUP BY 1 ORDER BY user_months DESC")
    show("membership_in_force_eom by month (last 6 months)", """
        SELECT month_start, SUM(membership_in_force_eom::INT) AS in_force,
               ROUND(AVG(normalized_revenue),2) AS avg_monthly_revenue
        FROM fct GROUP BY 1 ORDER BY 1 DESC LIMIT 6""")

    print(f"\nAll Phase 3 (v3) assertions passed for {label}.")


build_cohort(
    label="Feb (train.csv)", dim_name="dim_member_feb", as_of="2017-01-31",
    feature_tx_sql="SELECT * FROM tx_v1 WHERE transaction_date <= DATE '2017-01-31'",
    out_name="fct_member_month_feb",
)

build_cohort(
    label="Mar (train_v2.csv)", dim_name="dim_member_mar", as_of="2017-02-28",
    feature_tx_sql="""
        SELECT msno, transaction_date, membership_expire_date, payment_plan_days,
               plan_list_price, actual_amount_paid, is_auto_renew, is_cancel
        FROM tx_v1 WHERE transaction_date <= DATE '2017-02-28'
        UNION ALL
        SELECT msno, transaction_date, membership_expire_date, payment_plan_days,
               plan_list_price, actual_amount_paid, is_auto_renew, is_cancel
        FROM tx_v2 WHERE transaction_date <= DATE '2017-02-28'
    """,
    out_name="fct_member_month_mar",
)

print("\nBoth cohorts built. Phase 3 (v3) gate: OK to proceed to Phase 4.")