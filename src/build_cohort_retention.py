"""
build_cohort_retention.py  -  Phase 4 (v3): cohort retention, the way the KKBox literature defines it

Definition (matches the Databricks KKBox write-up and the pycox/Time-to-Event paper):
  * Churn   = KKBox's own rule: a membership has lapsed and the user did NOT pay again within 30 days
              of the running coverage end.  First such event only (a later return is "resurrection",
              counted but ignored for retention).
  * Clock   = starts at the user's FIRST transaction (not registration: members cover only ~88% of users).
  * Cohort  = month of first transaction.
  * Retained at month k  <=>  survival time >= 30*k days.  (30-day blocks, not calendar months, so edge
              effects of month length don't distort the curve.)  Retention can only fall as k grows.

Population: ALL users in transactions.csv (v1), not just train users.  Train users are all alive at the
as-of date by construction, so a heatmap built from them would be survivorship-biased.

Left truncation: transactions begin 2015-01-01, so the 2015-01 "first transaction" cohort is really
"already subscribed before the file starts".  It is flagged and excluded from the heatmap.

Right censoring: file ends 2017-02-28.  A cell (cohort, k) is shown only if EVERY user in that cohort
is fully observable at month k, i.e. last day of cohort month + 30*k + 30 days <= 2017-02-28.
(The extra 30 days is because a lapse is only knowable 30 days after expiry.)

Reads : data/staging/transactions_v1.parquet
Writes: data/core/user_survival.parquet      (one row per user: duration_days, event - reused by Phase 5)
        data/core/cohort_retention.parquet   (tidy long: cohort_month, k, cohort_size, retained, retention_rate)
"""
import os
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")

CENSOR = "2017-02-28"      # transactions.csv max date (asserted below)
GAP_DAYS = 30              # KKBox's churn rule
MAX_K = 25                 # 24 months of cohorts at most; 25 gives one spare column

os.makedirs(P("data", "core"), exist_ok=True)
con = duckdb.connect()
con.execute(f"CREATE VIEW tx AS SELECT msno, transaction_date, membership_expire_date, actual_amount_paid "
            f"FROM read_parquet('{P('data','staging','transactions_v1.parquet')}')")

max_date = con.execute("SELECT MAX(transaction_date) FROM tx").fetchone()[0]
assert str(max_date)[:10] == CENSOR, f"max transaction date is {max_date}, expected {CENSOR}"

# ---------------------------------------------------------------- user-level survival table
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ordered AS
    SELECT msno, transaction_date, membership_expire_date,
           MAX(membership_expire_date) OVER (
               PARTITION BY msno ORDER BY transaction_date, membership_expire_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_cover_end
    FROM tx
""")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE per_user AS
    SELECT msno,
           MIN(transaction_date)       AS first_tx_date,
           MAX(membership_expire_date) AS last_cover_end,
           COUNT(*)                    AS n_transactions,
           -- first gap > 30 days between the coverage so far and the next payment = first churn event
           MIN(CASE WHEN prev_cover_end IS NOT NULL
                     AND transaction_date > prev_cover_end + {GAP_DAYS}
                    THEN prev_cover_end END) AS first_gap_churn_date,
           COUNT(CASE WHEN prev_cover_end IS NOT NULL
                       AND transaction_date > prev_cover_end + {GAP_DAYS} THEN 1 END) AS n_gaps
    FROM ordered
    GROUP BY msno
""")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE user_survival AS
    SELECT msno, first_tx_date,
           CAST(DATE_TRUNC('month', first_tx_date) AS DATE) AS cohort_month,
           (CAST(DATE_TRUNC('month', first_tx_date) AS DATE) = DATE '2015-01-01') AS is_left_truncated_cohort,
           n_transactions, n_gaps,
           -- churn date: first gap if there is one; else the final lapse if it is already knowable
           CASE WHEN first_gap_churn_date IS NOT NULL THEN first_gap_churn_date
                WHEN last_cover_end + {GAP_DAYS} <= DATE '{CENSOR}' THEN last_cover_end
           END AS churn_date,
           (first_gap_churn_date IS NOT NULL
            OR last_cover_end + {GAP_DAYS} <= DATE '{CENSOR}') AS event
    FROM per_user
""")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE user_survival2 AS
    SELECT *,
           DATE_DIFF('day', first_tx_date, COALESCE(churn_date, DATE '{CENSOR}')) AS raw_duration_days,
           GREATEST(0, DATE_DIFF('day', first_tx_date, COALESCE(churn_date, DATE '{CENSOR}'))) AS duration_days
    FROM user_survival
""")

n_users, n_events, n_cens, n_clipped, n_resurrect = con.execute("""
    SELECT COUNT(*), SUM(event::INT), SUM((NOT event)::INT),
           SUM((raw_duration_days < 0)::INT),
           SUM((n_gaps > 0 AND (churn_date IS NULL OR n_gaps > 1))::INT)
    FROM user_survival2""").fetchone()

print(f"Users: {n_users:,}   churn events: {n_events:,} ({100*n_events/n_users:.1f}%)   censored: {n_cens:,}")
print(f"Users with a gap>30d that later returned or lapsed again (resurrection): {n_resurrect:,}")
print(f"Users with negative raw duration (expiry before payment date), clipped to 0: {n_clipped:,}")

# gates on the user table
assert n_users == con.execute("SELECT COUNT(DISTINCT msno) FROM tx").fetchone()[0], "user count mismatch"
assert con.execute("SELECT COUNT(*) FROM (SELECT msno FROM user_survival2 GROUP BY 1 HAVING COUNT(*)>1)").fetchone()[0] == 0
# Negative durations were investigated (inspect_negative_durations.py): every one of them has a transaction row whose
# membership_expire_date is BEFORE its transaction_date, and ~95% of such rows are is_cancel=1 cancellation records.
# So they are explained, not a bug.  We keep KKBox's official expiry dates untouched (consistent with the Phase 0 label
# reconstruction) and clip the duration at 0.  Clipping cannot change any retention cell: a duration of 0 or -2 is
# equally "< 30 days".  The gate below is therefore: every negative user must be explained by such a row.
unexplained = con.execute("""
    SELECT COUNT(*) FROM user_survival2
    WHERE raw_duration_days < 0
      AND msno NOT IN (SELECT msno FROM tx WHERE membership_expire_date < transaction_date)""").fetchone()[0]
assert unexplained == 0, f"{unexplained:,} negative-duration users have NO expiry-before-payment row - unexplained, investigate"
print(f"All {n_clipped:,} negative durations ({100*n_clipped/n_users:.2f}% of users) are explained by cancel-type rows; clipped to 0.")

# first-payment type: a FREE first payment (actual_amount_paid = 0, i.e. a trial/promo) vs a PAID one.  Found in
# inspect_first_payment_by_cohort.py: cohorts like 2015-06 / 2016-10 / 2016-11 are 70-86% free-first and ~95% of those
# users lapse inside 30 days, so mixing the two populations makes retention look like it is collapsing when paid-user
# retention is actually stable.  Known on day 0, so it is also a legitimate Phase 6 feature.
con.execute("""
    CREATE OR REPLACE TEMP TABLE user_survival3 AS
    SELECT u.*, CASE WHEN t.actual_amount_paid = 0 THEN 'free_first' ELSE 'paid_first' END AS first_payment_type
    FROM user_survival2 u JOIN tx t ON t.msno = u.msno AND t.transaction_date = u.first_tx_date""")
n3 = con.execute("SELECT COUNT(*) FROM user_survival3").fetchone()[0]
assert n3 == n_users, f"first-payment join changed the user count: {n3:,} vs {n_users:,}"

con.execute(f"COPY (SELECT msno, first_tx_date, cohort_month, is_left_truncated_cohort, first_payment_type, "
            f"n_transactions, n_gaps, churn_date, event, duration_days FROM user_survival3) "
            f"TO '{P('data','core','user_survival.parquet')}' (FORMAT PARQUET)")
print(f"Wrote {P('data','core','user_survival.parquet')}")

# ---------------------------------------------------------------- cohort x k table, by segment
# segments: 'all', 'paid_first', 'free_first'
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cohort_retention AS
    WITH base AS (
        SELECT cohort_month, duration_days, 'all' AS segment
        FROM user_survival3 WHERE NOT is_left_truncated_cohort
        UNION ALL
        SELECT cohort_month, duration_days, first_payment_type
        FROM user_survival3 WHERE NOT is_left_truncated_cohort
    ),
    cohorts AS (SELECT segment, cohort_month, COUNT(*) AS cohort_size FROM base GROUP BY 1, 2),
    ks AS (SELECT UNNEST(generate_series(0, {MAX_K})) AS k),
    grid AS (
        SELECT c.segment, c.cohort_month, c.cohort_size, ks.k
        FROM cohorts c CROSS JOIN ks
        -- show a cell only when every user in the cohort is fully observable at month k
        WHERE CAST(c.cohort_month AS DATE) + INTERVAL 1 MONTH - INTERVAL 1 DAY + (30*ks.k + {GAP_DAYS}) * INTERVAL 1 DAY
              <= DATE '{CENSOR}'
    )
    SELECT g.segment, g.cohort_month, g.k, g.cohort_size,
           SUM((b.duration_days >= 30*g.k)::INT) AS retained
    FROM grid g
    JOIN base b ON b.segment = g.segment AND b.cohort_month = g.cohort_month
    GROUP BY g.segment, g.cohort_month, g.k, g.cohort_size
""")
con.execute("CREATE OR REPLACE TEMP TABLE cr AS SELECT *, retained::DOUBLE / cohort_size AS retention_rate FROM cohort_retention")

# ---------------------------------------------------------------- gates
bad0 = con.execute("SELECT COUNT(*) FROM cr WHERE k = 0 AND retained <> cohort_size").fetchone()[0]
assert bad0 == 0, f"{bad0} segment-cohorts are not 100% retained at k=0"

rises = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT segment, cohort_month, k, retained,
               LAG(retained) OVER (PARTITION BY segment, cohort_month ORDER BY k) AS prev
        FROM cr) WHERE prev IS NOT NULL AND retained > prev
""").fetchone()[0]
assert rises == 0, f"{rises} cells where retention rose with k - not a survival curve"

# the two segments must partition 'all', cohort by cohort
mismatch = con.execute("""
    WITH s AS (SELECT DISTINCT segment, cohort_month, cohort_size FROM cr)
    SELECT COUNT(*) FROM (
        SELECT a.cohort_month, a.cohort_size AS all_n,
               COALESCE(p.cohort_size, 0) + COALESCE(f.cohort_size, 0) AS parts_n
        FROM s a
        LEFT JOIN s p ON p.cohort_month = a.cohort_month AND p.segment = 'paid_first'
        LEFT JOIN s f ON f.cohort_month = a.cohort_month AND f.segment = 'free_first'
        WHERE a.segment = 'all') WHERE all_n <> parts_n
""").fetchone()[0]
assert mismatch == 0, f"{mismatch} cohorts where paid_first + free_first <> all"

tot = con.execute("SELECT SUM(cohort_size) FROM (SELECT DISTINCT cohort_month, cohort_size FROM cr WHERE segment='all')").fetchone()[0]
n_cohorts = con.execute("SELECT COUNT(DISTINCT cohort_month) FROM cr").fetchone()[0]
n_trunc = con.execute("SELECT COUNT(*) FROM user_survival3 WHERE is_left_truncated_cohort").fetchone()[0]
print(f"Cohorts shown: {n_cohorts}  |  users in them: {tot:,}  |  excluded 2015-01 (left-truncated) users: {n_trunc:,}")
seg_users = con.execute("""
    SELECT segment, SUM(cohort_size) AS users FROM (SELECT DISTINCT segment, cohort_month, cohort_size FROM cr)
    GROUP BY 1 ORDER BY 1""").df()
print(seg_users.to_string(index=False))

con.execute(f"COPY (SELECT segment, cohort_month, k, cohort_size, retained, retention_rate FROM cr ORDER BY 1, 2, 3) "
            f"TO '{P('data','core','cohort_retention.parquet')}' (FORMAT PARQUET)")
print(f"Wrote {P('data','core','cohort_retention.parquet')}")

# ---------------------------------------------------------------- readable summary
show_k_wanted = (0, 1, 2, 3, 6, 12, 18)
for seg in ("paid_first", "free_first"):
    df = con.execute(f"SELECT cohort_month, k, retention_rate, cohort_size FROM cr WHERE segment = '{seg}'").df()
    tri = df.pivot(index="cohort_month", columns="k", values="retention_rate")
    sizes = df.drop_duplicates("cohort_month").set_index("cohort_month")["cohort_size"]
    show_k = [k for k in show_k_wanted if k in tri.columns]
    out = (tri[show_k] * 100).round(1)
    out.insert(0, "size", sizes)
    out.index = [str(i)[:7] for i in out.index]
    print(f"\nRetention % - segment {seg} (rows = cohort, columns = month k since first payment):")
    print(out.to_string())

print("\nAll Phase 4 (v3) assertions passed.")