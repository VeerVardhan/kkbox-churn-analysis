"""
src/inspect_label_window.py

Diagnostic - NOT a pipeline step. Tests one hypothesis before we write build_dim_member.py:

  H1: train_v2 = users whose membership, AS KNOWN ON 2017-03-01, expires during March 2017.
      Their label (renewed within 30 days of that expiry, or not) is mostly decided by events
      on/after 2017-03-01 - so a leakage-safe feature cutoff is the as-of date, and
      "strictly before the expiry date" (Decision 6 as worded) may NOT be strict enough
      because an EARLY renewal (transaction dated before the expiry, with a later expiry) is
      itself the label.

Reads data/staging/*.parquet only. Prints a report and saves data/interim/label_window_report.txt.
"""
import io
import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(PROJECT_ROOT, "data", "staging")
INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
os.makedirs(INTERIM, exist_ok=True)

REF = "2017-03-01"  # hypothesised "as-of" date

buf = io.StringIO()


def say(text=""):
    print(text)
    buf.write(str(text) + "\n")


def parquet(name):
    return os.path.join(STAGING, name + ".parquet").replace("\\", "/")


con = duckdb.connect()
con.execute(f"CREATE VIEW train AS SELECT * FROM read_parquet('{parquet('train')}')")
con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{parquet('transactions')}')")


def q(title, sql):
    say(f"\n--- {title}")
    say(con.execute(sql).df().to_string(index=False))


# One row per train user: the expiry as known on REF (using only transactions dated before REF).
con.execute(f"""
CREATE VIEW user_asof AS
SELECT tr.msno,
       tr.is_churn,
       MAX(CASE WHEN t.transaction_date < DATE '{REF}' THEN t.membership_expire_date END) AS expiry_asof,
       COUNT(t.msno) AS n_tx_total,
       COALESCE(SUM(CASE WHEN t.transaction_date < DATE '{REF}' THEN 1 ELSE 0 END), 0) AS n_tx_before_ref
FROM train tr
LEFT JOIN tx t ON t.msno = tr.msno
GROUP BY tr.msno, tr.is_churn
""")

say("=" * 70)
say(f"LABEL-WINDOW DIAGNOSTIC (hypothesised as-of date {REF})")
say("=" * 70)

q("Q1. Month of the expiry known as of the ref date (H1 predicts March 2017 dominates)", """
SELECT COALESCE(CAST(DATE_TRUNC('month', expiry_asof) AS VARCHAR), '<no tx before ref>') AS expiry_asof_month,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(is_churn), 1) AS churn_pct
FROM user_asof
GROUP BY 1
ORDER BY users DESC
LIMIT 15
""")

q("Q2. Users with as-of expiry in March 2017: how many transactions before the ref date?", """
SELECT LEAST(n_tx_before_ref, 4) AS n_tx_before_ref_capped_at_4,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(is_churn), 1) AS churn_pct
FROM user_asof
WHERE expiry_asof BETWEEN DATE '2017-03-01' AND DATE '2017-03-31'
GROUP BY 1
ORDER BY 1
""")

con.execute(f"""
CREATE VIEW march AS
SELECT b.msno, b.is_churn, b.expiry_asof,
       MAX(CASE WHEN t.transaction_date >= DATE '{REF}'
                 AND t.membership_expire_date > b.expiry_asof THEN 1 ELSE 0 END) AS extended_in_file,
       MAX(CASE WHEN t.transaction_date >= DATE '{REF}'
                 AND t.membership_expire_date > b.expiry_asof
                 AND t.transaction_date <  b.expiry_asof THEN 1 ELSE 0 END) AS extended_early,
       MAX(CASE WHEN t.transaction_date >= DATE '{REF}'
                 AND t.membership_expire_date > b.expiry_asof
                 AND t.transaction_date >= b.expiry_asof THEN 1 ELSE 0 END) AS extended_on_or_after
FROM user_asof b
LEFT JOIN tx t ON t.msno = b.msno
WHERE b.expiry_asof BETWEEN DATE '2017-03-01' AND DATE '2017-03-31'
GROUP BY b.msno, b.is_churn, b.expiry_asof
""")

q("Q3. March-expiry users: did a transaction on/after the ref date push the expiry later? (the label in disguise)", """
SELECT extended_in_file, extended_early, extended_on_or_after,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(is_churn), 1) AS churn_pct
FROM march
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
""")

q("Q4. March-expiry users NOT extended in the file: churn % by week of expiry "
  "(later expiry = less of the 30-day window is inside the file)", """
SELECT CAST(DATE_TRUNC('week', expiry_asof) AS VARCHAR) AS expiry_week,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(is_churn), 1) AS churn_pct
FROM march
WHERE extended_in_file = 0
GROUP BY 1
ORDER BY 1
""")

q("Q5. Transactions of train users by month (is history really thin?)", """
SELECT CAST(DATE_TRUNC('month', transaction_date) AS VARCHAR) AS tx_month, COUNT(*) AS rows_
FROM tx
WHERE msno IN (SELECT msno FROM train)
GROUP BY 1
ORDER BY 1
""")

q("Q6. Train users with no usable pre-ref history", """
SELECT CASE WHEN n_tx_total = 0 THEN 'no transactions at all'
            WHEN n_tx_before_ref = 0 THEN 'transactions only on/after ref'
            ELSE 'has pre-ref transactions' END AS group_,
       COUNT(*) AS users,
       ROUND(100.0 * AVG(is_churn), 1) AS churn_pct
FROM user_asof
GROUP BY 1
ORDER BY users DESC
""")

out = os.path.join(INTERIM, "label_window_report.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
say(f"\nReport saved to {out}")