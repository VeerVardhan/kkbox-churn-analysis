"""
inspect_negative_durations.py  -  why do 12,050 users have churn_date before their first payment?  (read-only)

Hypothesis H1: some transaction rows have membership_expire_date < transaction_date (expiry before the
               payment that created the membership) - junk rows, and the negative durations come from them.
We test that, characterise the rows, and measure what a principled repair would change:
    cover_end = GREATEST(membership_expire_date, transaction_date)
(a membership cannot end before it starts).

Reads: data/staging/transactions_v1.parquet     Writes: nothing
"""
import os
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")
CENSOR, GAP = "2017-02-28", 30

con = duckdb.connect()
con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{P('data','staging','transactions_v1.parquet')}')")


def show(title, sql):
    print(f"\n--- {title}")
    print(con.execute(sql).df().to_string(index=False))


def build_survival(name, cover_expr):
    """Same logic as build_cohort_retention.py, with a switchable definition of coverage end."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE o_{name} AS
        SELECT msno, transaction_date, {cover_expr} AS cover_end,
               MAX({cover_expr}) OVER (PARTITION BY msno ORDER BY transaction_date, membership_expire_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_cover_end
        FROM tx""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s_{name} AS
        SELECT msno, MIN(transaction_date) AS first_tx_date,
               MAX(cover_end) AS last_cover_end,
               MIN(CASE WHEN prev_cover_end IS NOT NULL AND transaction_date > prev_cover_end + {GAP}
                        THEN prev_cover_end END) AS gap_churn
        FROM o_{name} GROUP BY msno""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE u_{name} AS
        SELECT msno, first_tx_date, gap_churn, last_cover_end,
               CASE WHEN gap_churn IS NOT NULL THEN gap_churn
                    WHEN last_cover_end + {GAP} <= DATE '{CENSOR}' THEN last_cover_end END AS churn_date,
               (gap_churn IS NOT NULL OR last_cover_end + {GAP} <= DATE '{CENSOR}') AS event,
               DATE_DIFF('day', first_tx_date,
                         COALESCE(CASE WHEN gap_churn IS NOT NULL THEN gap_churn
                                       WHEN last_cover_end + {GAP} <= DATE '{CENSOR}' THEN last_cover_end END,
                                  DATE '{CENSOR}')) AS raw_dur
        FROM s_{name}""")


build_survival("raw", "membership_expire_date")
build_survival("fix", "GREATEST(membership_expire_date, transaction_date)")

# ------------------------------------------------------------------ 1. row level
show("1. Row-level: transactions whose expiry is before the transaction date", f"""
    SELECT COUNT(*) AS rows_total,
           SUM((membership_expire_date < transaction_date)::INT) AS expiry_before_tx,
           ROUND(100.0*SUM((membership_expire_date < transaction_date)::INT)/COUNT(*), 3) AS pct,
           MIN(DATE_DIFF('day', transaction_date, membership_expire_date)) AS min_days_to_expiry
    FROM tx""")

show("1b. Those rows by plan length and amount", """
    SELECT CASE WHEN payment_plan_days <= 0 THEN 'plan<=0' WHEN payment_plan_days <= 31 THEN 'monthly-ish'
                ELSE 'longer' END AS plan,
           (actual_amount_paid = 0) AS free, is_cancel, COUNT(*) AS n
    FROM tx WHERE membership_expire_date < transaction_date
    GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 12""")

# ------------------------------------------------------------------ 2. the 12,050 users
show("2. Negative-duration users: do they have an expiry-before-tx row?", """
    WITH neg AS (SELECT msno FROM u_raw WHERE raw_dur < 0),
         bad AS (SELECT DISTINCT msno FROM tx WHERE membership_expire_date < transaction_date)
    SELECT COUNT(*) AS neg_users,
           SUM((msno IN (SELECT msno FROM bad))::INT) AS with_junk_row,
           SUM((msno NOT IN (SELECT msno FROM bad))::INT) AS without_junk_row
    FROM neg""")

show("2b. Negative-duration users: which branch produced the churn date, and how negative?", """
    SELECT CASE WHEN gap_churn IS NOT NULL THEN 'first gap' ELSE 'final lapse' END AS branch,
           COUNT(*) AS users, MIN(raw_dur) AS most_negative, ROUND(MEDIAN(raw_dur), 0) AS median_dur
    FROM u_raw WHERE raw_dur < 0 GROUP BY 1""")

show("2c. Negative-duration users: number of transactions", """
    SELECT LEAST(c, 5) AS n_tx_capped, COUNT(*) AS users
    FROM (SELECT msno, COUNT(*) AS c FROM tx WHERE msno IN (SELECT msno FROM u_raw WHERE raw_dur < 0) GROUP BY 1)
    GROUP BY 1 ORDER BY 1""")

show("2d. Five example users, all their rows", """
    SELECT msno, transaction_date, membership_expire_date, payment_plan_days, actual_amount_paid, is_cancel
    FROM tx WHERE msno IN (SELECT msno FROM u_raw WHERE raw_dur < 0 LIMIT 5)
    ORDER BY msno, transaction_date""")

# ------------------------------------------------------------------ 3. what does the repair change?
show("3. Repair GREATEST(expiry, transaction_date): negatives left, and what changes vs raw", """
    SELECT (SELECT COUNT(*) FROM u_fix WHERE raw_dur < 0) AS negatives_after_fix,
           SUM((r.event <> f.event)::INT) AS users_whose_event_flag_changes,
           SUM((r.event AND f.event AND r.churn_date <> f.churn_date)::INT) AS users_whose_churn_date_changes,
           SUM((r.event)::INT) AS events_raw, SUM((f.event)::INT) AS events_fixed
    FROM u_raw r JOIN u_fix f USING (msno)""")

show("3b. Same, restricted to users that were negative in raw", """
    SELECT COUNT(*) AS users,
           SUM((r.event <> f.event)::INT) AS event_flips,
           SUM((r.churn_date <> f.churn_date)::INT) AS date_changes,
           ROUND(MEDIAN(f.raw_dur), 0) AS median_fixed_duration
    FROM u_raw r JOIN u_fix f USING (msno) WHERE r.raw_dur < 0""")