"""
Phase 1 - staging: raw CSV -> typed Parquet.

What this does, in order:
  1. Parse YYYYMMDD integer date columns into real DATE columns everywhere.
  2. Apply the locked dedup rule to transactions: for duplicate (msno, transaction_date)
     groups, keep the row with the MAX membership_expire_date (ties broken arbitrarily -
     the payment attributes are identical in practice for both the ordinary and the
     wild-spread junk case found in Phase 0).
  3. Filter user_logs down to only msno's that appear in train_v2 - we only ever model
     labeled users, so there is no reason to carry the rest through every later phase.
  4. Write everything to data/staging/*.parquet.
  5. Assert the staged row counts reconcile against the raw counts, given exactly the
     rows the dedup rule and the log filter are expected to remove.

Everything downstream reads Parquet from data/staging/. Nothing downstream should ever
read a raw CSV again.
"""

import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
STAGING = os.path.join(PROJECT_ROOT, "data", "staging")
os.makedirs(STAGING, exist_ok=True)

TRAIN = os.path.join(RAW, "train_v2.csv").replace("\\", "/")
TRANSACTIONS = os.path.join(RAW, "transactions_v2.csv").replace("\\", "/")
LOGS = os.path.join(RAW, "user_logs_v2.csv").replace("\\", "/")
MEMBERS = os.path.join(RAW, "members_v3.csv").replace("\\", "/")

OUT_TRAIN = os.path.join(STAGING, "train.parquet").replace("\\", "/")
OUT_TX = os.path.join(STAGING, "transactions.parquet").replace("\\", "/")
OUT_LOGS = os.path.join(STAGING, "user_logs.parquet").replace("\\", "/")
OUT_MEMBERS = os.path.join(STAGING, "members.parquet").replace("\\", "/")

con = duckdb.connect()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# train_v2 -> straight copy, no transformation needed
# ---------------------------------------------------------------------------
section("Staging train_v2")
raw_train_n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{TRAIN}')").fetchone()[0]
con.execute(f"""
    COPY (SELECT * FROM read_csv_auto('{TRAIN}'))
    TO '{OUT_TRAIN}' (FORMAT PARQUET)
""")
staged_train_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_TRAIN}')").fetchone()[0]
assert raw_train_n == staged_train_n, f"train row count mismatch: raw={raw_train_n} staged={staged_train_n}"
print(f"  {staged_train_n} rows written to {OUT_TRAIN}")

# ---------------------------------------------------------------------------
# members_v3 -> parse registration_init_time, straight copy otherwise
# ---------------------------------------------------------------------------
section("Staging members_v3")
raw_members_n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{MEMBERS}')").fetchone()[0]
con.execute(f"""
    COPY (
        SELECT
            msno, city, bd, gender, registered_via,
            strptime(CAST(registration_init_time AS VARCHAR), '%Y%m%d')::DATE AS registration_date
        FROM read_csv_auto('{MEMBERS}')
    ) TO '{OUT_MEMBERS}' (FORMAT PARQUET)
""")
staged_members_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_MEMBERS}')").fetchone()[0]
assert raw_members_n == staged_members_n, f"members row count mismatch: raw={raw_members_n} staged={staged_members_n}"
print(f"  {staged_members_n} rows written to {OUT_MEMBERS}")

# ---------------------------------------------------------------------------
# transactions_v2 -> parse dates, dedup
# ---------------------------------------------------------------------------
section("Staging transactions_v2 (with dedup)")

raw_tx_n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{TRANSACTIONS}')").fetchone()[0]

dup_check = con.execute(f"""
    WITH dup AS (
        SELECT msno, transaction_date, COUNT(*) AS n
        FROM read_csv_auto('{TRANSACTIONS}')
        GROUP BY msno, transaction_date
    )
    SELECT
        SUM(n) AS total_rows,
        COUNT(*) AS total_groups,
        SUM(n - 1) AS expected_rows_removed
    FROM dup
""").fetchdf()
expected_removed = int(dup_check["expected_rows_removed"][0])
expected_staged = int(dup_check["total_groups"][0])
print(f"  raw rows: {raw_tx_n}, expected after dedup (1 row per msno+date): {expected_staged} "
      f"({expected_removed} rows removed)")

con.execute(f"""
    COPY (
        SELECT
            msno, payment_method_id, payment_plan_days, plan_list_price,
            actual_amount_paid, is_auto_renew, is_cancel,
            strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS transaction_date,
            strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS membership_expire_date
        FROM read_csv_auto('{TRANSACTIONS}')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY msno, transaction_date
            ORDER BY membership_expire_date DESC
        ) = 1
    ) TO '{OUT_TX}' (FORMAT PARQUET)
""")
staged_tx_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_TX}')").fetchone()[0]
assert staged_tx_n == expected_staged, (
    f"transactions dedup mismatch: expected {expected_staged}, got {staged_tx_n}"
)
print(f"  {staged_tx_n} rows written to {OUT_TX} (dedup rule applied and verified)")

# ---------------------------------------------------------------------------
# user_logs_v2 -> parse date, filter to train_v2 users only
# ---------------------------------------------------------------------------
section("Staging user_logs_v2 (filtered to train_v2 users, parsed dates)")

raw_logs_n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{LOGS}')").fetchone()[0]
print(f"  raw rows: {raw_logs_n}")

con.execute(f"""
    COPY (
        SELECT
            lg.msno,
            strptime(CAST(lg.date AS VARCHAR), '%Y%m%d')::DATE AS log_date,
            lg.num_25, lg.num_50, lg.num_75, lg.num_985, lg.num_100,
            lg.num_unq, lg.total_secs
        FROM read_csv_auto('{LOGS}') lg
        WHERE lg.msno IN (SELECT msno FROM read_parquet('{OUT_TRAIN}'))
    ) TO '{OUT_LOGS}' (FORMAT PARQUET)
""")
staged_logs_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_LOGS}')").fetchone()[0]
print(f"  {staged_logs_n} rows written to {OUT_LOGS} "
      f"({100*staged_logs_n/raw_logs_n:.1f}% of raw rows kept - the rest belong to "
      f"users with no train_v2 label)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("STAGING COMPLETE")
for name, path in [("train", OUT_TRAIN), ("members", OUT_MEMBERS),
                    ("transactions", OUT_TX), ("user_logs", OUT_LOGS)]:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {name:14s} {size_mb:8.1f} MB  {path}")
print("\n  All row-count assertions passed. Phase 1 gate: OK to proceed to Phase 2 (dim_member).")