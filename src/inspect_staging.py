"""
src/inspect_staging.py

Phase 1 gate check + Phase 2 pre-flight.
Reads ONLY data/staging/*.parquet (never the raw CSVs). Prints a report and saves it to
data/interim/staging_preflight_report.txt.

Part A  - Phase 1 gate: types, uniqueness, date ranges (is staging what build_staging.py promised?)
Part B  - Phase 2 pre-flight: facts we need BEFORE writing build_dim_member.py
          (age values, gender nulls, registration dates, expiry-date shape, log outliers).
It makes no decisions and writes no data - it only measures.
"""
import io
import os
import sys
import glob
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(PROJECT_ROOT, "data", "staging")
INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
os.makedirs(INTERIM, exist_ok=True)

buf = io.StringIO()


def say(text=""):
    print(text)
    buf.write(str(text) + "\n")


def find_file(keyword):
    hits = [p for p in glob.glob(os.path.join(STAGING, "*.parquet")) if keyword in os.path.basename(p).lower()]
    if len(hits) != 1:
        sys.exit(f"Expected exactly one parquet containing '{keyword}' in {STAGING}, found: {hits}")
    return hits[0].replace("\\", "/")


con = duckdb.connect()
paths = {k: find_file(k) for k in ["train", "members", "transactions", "user_logs"]}
for name, p in paths.items():
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p}')")


con.execute("CREATE VIEW members_train AS SELECT * FROM members WHERE msno IN (SELECT msno FROM train)")


def q(title, sql):
    say(f"\n--- {title}")
    df = con.execute(sql).df()
    say(df.to_string(index=False))
    return df


# ------------------------------------------------------------------ Part A
say("=" * 70)
say("PART A - PHASE 1 GATE (staging integrity)")
say("=" * 70)

for name, p in paths.items():
    size_mb = os.path.getsize(p) / 1e6
    n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    say(f"\n{name}: {n:,} rows, {size_mb:,.1f} MB  ({os.path.basename(p)})")
    say(con.execute(f"DESCRIBE {name}").df()[["column_name", "column_type"]].to_string(index=False))

q("A1. Uniqueness of natural keys (every 'dupes' value must be 0)", """
SELECT 'train.msno' AS key, COUNT(*) - COUNT(DISTINCT msno) AS dupes FROM train
UNION ALL SELECT 'members.msno', COUNT(*) - COUNT(DISTINCT msno) FROM members
UNION ALL SELECT 'transactions.(msno,transaction_date)',
       COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT msno, transaction_date FROM transactions)) FROM transactions
UNION ALL SELECT 'user_logs.(msno,log_date)',
       COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT msno, log_date FROM user_logs)) FROM user_logs
""")

q("A2. Date ranges (transactions.max should be 2017-03-31)", """
SELECT 'transaction_date' AS col, MIN(transaction_date) AS min_d, MAX(transaction_date) AS max_d FROM transactions
UNION ALL SELECT 'membership_expire_date', MIN(membership_expire_date), MAX(membership_expire_date) FROM transactions
UNION ALL SELECT 'registration_date', MIN(registration_date), MAX(registration_date) FROM members
UNION ALL SELECT 'logs.date', MIN(log_date), MAX(log_date) FROM user_logs
""")

q("A3. Orphans: log users not in train (must be 0, Decision 9)", """
SELECT COUNT(DISTINCT msno) AS log_users_not_in_train
FROM user_logs WHERE msno NOT IN (SELECT msno FROM train)
""")

q("A4. Coverage of train users (should match Phase 0: 96.1 / 88.7 / 77.7)", """
SELECT
  ROUND(100.0 * COUNT(DISTINCT t.msno)  / (SELECT COUNT(*) FROM train), 1) AS pct_in_transactions,
  ROUND(100.0 * COUNT(DISTINCT m.msno)  / (SELECT COUNT(*) FROM train), 1) AS pct_in_members,
  ROUND(100.0 * COUNT(DISTINCT l.msno)  / (SELECT COUNT(*) FROM train), 1) AS pct_in_logs
FROM train tr
LEFT JOIN (SELECT DISTINCT msno FROM transactions) t USING (msno)
LEFT JOIN members m USING (msno)
LEFT JOIN (SELECT DISTINCT msno FROM user_logs) l USING (msno)
""")

# ------------------------------------------------------------------ Part B
say("\n" + "=" * 70)
say("PART B - PHASE 2 PRE-FLIGHT (facts needed before build_dim_member.py)")
say("=" * 70)

q("B1. Age (bd), train users only - known to hold impossible values; shows what a flag rule has to catch", """
SELECT
  COUNT(*)                                        AS members,
  SUM((bd IS NULL)::INT)                          AS null_bd,
  SUM((bd <= 0)::INT)                             AS zero_or_negative,
  SUM((bd BETWEEN 1 AND 9)::INT)                  AS age_1_to_9,
  SUM((bd BETWEEN 10 AND 80)::INT)                AS age_10_to_80,
  SUM((bd BETWEEN 81 AND 100)::INT)               AS age_81_to_100,
  SUM((bd > 100)::INT)                            AS above_100,
  MIN(bd) AS min_bd, MAX(bd) AS max_bd
FROM members_train
""")

q("B2. (train users) Gender values (nulls ~60% per Phase 0) and registered_via values", """
SELECT 'gender' AS col, COALESCE(CAST(gender AS VARCHAR), '<NULL>') AS value, COUNT(*) AS n FROM members_train GROUP BY 1, 2
UNION ALL
SELECT 'registered_via', COALESCE(CAST(registered_via AS VARCHAR), '<NULL>'), COUNT(*) FROM members_train GROUP BY 1, 2
ORDER BY col, n DESC
""")

q("B3. City: distinct values and top 5 (decides whether city needs bucketing before Phase 4 segments)", """
SELECT COUNT(DISTINCT city) AS distinct_cities FROM members_train
""")
q("B3b. Top 5 cities", """
SELECT city, COUNT(*) AS n FROM members_train GROUP BY 1 ORDER BY n DESC LIMIT 5
""")

q("B4. Registration month of train members (cohort sizes - drives monthly vs quarterly in Phase 4)", """
SELECT DATE_TRUNC('year', registration_date) AS reg_year, COUNT(*) AS members
FROM members_train
GROUP BY 1 ORDER BY 1
""")

q("B5. Registration date sanity vs first transaction (negative tenure risk)", """
WITH first_tx AS (SELECT msno, MIN(transaction_date) AS first_tx FROM transactions GROUP BY msno)
SELECT
  COUNT(*) AS train_users_with_member_and_tx,
  SUM((m.registration_date > f.first_tx)::INT) AS registered_AFTER_first_tx,
  SUM((m.registration_date > DATE '2017-03-31')::INT) AS registered_after_cutoff
FROM train tr JOIN members m USING (msno) JOIN first_tx f USING (msno)
""")

q("B6. Shape of each train user's LAST membership_expire_date (which month do they land in?)", """
WITH last_exp AS (
  SELECT msno, MAX(membership_expire_date) AS last_expiry FROM transactions GROUP BY msno)
SELECT DATE_TRUNC('month', le.last_expiry) AS last_expiry_month,
       COUNT(*) AS users,
       ROUND(AVG(tr.is_churn) * 100, 1) AS churn_pct
FROM train tr JOIN last_exp le USING (msno)
GROUP BY 1 ORDER BY users DESC LIMIT 12
""")

q("B7. Transaction history: how many transactions per train user (feeds first/last/count columns)", """
SELECT COUNT(*) AS users,
       MIN(n) AS min_n, MEDIAN(n) AS median_n, QUANTILE_CONT(n, 0.99) AS p99_n, MAX(n) AS max_n
FROM (SELECT msno, COUNT(*) AS n FROM transactions WHERE msno IN (SELECT msno FROM train) GROUP BY msno)
""")

q("B8. Log outliers (Part 2 of the plan says verify these yourself)", """
SELECT
  COUNT(*) AS log_rows,
  SUM((total_secs < 0)::INT)      AS negative_secs,
  SUM((total_secs > 86400)::INT)  AS secs_over_one_day,
  SUM((num_unq < 0 OR num_25 < 0 OR num_50 < 0 OR num_75 < 0 OR num_985 < 0 OR num_100 < 0)::INT) AS negative_counts,
  MAX(total_secs) AS max_secs
FROM user_logs
""")

out_path = os.path.join(INTERIM, "staging_preflight_report.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
say(f"\nReport saved to {out_path}")