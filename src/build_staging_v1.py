"""
src/build_staging_v1.py  -  Phase 1 (v3): stage the real history files.

Converts train.csv and transactions.csv (the v1 files) to typed Parquet, applying the
same locked dedup rule (Decision 7) used on transactions_v2 in build_staging.py. Kept as
a separate script, rather than merged into build_staging.py, so each file's staging logic
stays easy to read and the original v2 staging run is never silently re-executed.

Reads : data/raw/train.csv, data/raw/transactions.csv
Writes: data/staging/train_feb.parquet, data/staging/transactions_v1.parquet
"""
import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
STAGING = os.path.join(PROJECT_ROOT, "data", "staging")
os.makedirs(STAGING, exist_ok=True)

TRAIN_RAW = os.path.join(RAW, "train.csv").replace("\\", "/")
TX_RAW = os.path.join(RAW, "transactions.csv").replace("\\", "/")
TRAIN_OUT = os.path.join(STAGING, "train_feb.parquet").replace("\\", "/")
TX_OUT = os.path.join(STAGING, "transactions_v1.parquet").replace("\\", "/")

for name, path in [("train.csv", TRAIN_RAW), ("transactions.csv", TX_RAW)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path} - check data/raw/")

con = duckdb.connect()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Stage train.csv -> train_feb.parquet
# ---------------------------------------------------------------------------
section("Staging train.csv (Feb cohort)")

con.execute(f"""
    COPY (SELECT * FROM read_csv_auto('{TRAIN_RAW}')) TO '{TRAIN_OUT}' (FORMAT PARQUET)
""")
n_train = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TRAIN_OUT}')").fetchone()[0]
n_train_raw = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{TRAIN_RAW}')").fetchone()[0]
assert n_train == n_train_raw, f"row count mismatch: staged {n_train} vs raw {n_train_raw}"
print(f"  {n_train:,} rows written to {TRAIN_OUT}")

# ---------------------------------------------------------------------------
# Stage transactions.csv -> transactions_v1.parquet (dedup applied, same rule as v2)
# ---------------------------------------------------------------------------
section("Staging transactions.csv (with dedup - Decision 7)")

n_raw = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{TX_RAW}')").fetchone()[0]
n_groups = con.execute(f"""
    SELECT COUNT(*) FROM (
        SELECT msno, transaction_date FROM read_csv_auto('{TX_RAW}') GROUP BY 1, 2
    )
""").fetchone()[0]
print(f"  raw rows: {n_raw}, expected after dedup (1 row per msno+date): {n_groups} "
      f"({n_raw - n_groups} rows removed)")

con.execute(f"""
    COPY (
        SELECT
            msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid,
            is_auto_renew,
            strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS transaction_date,
            strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS membership_expire_date,
            is_cancel
        FROM read_csv_auto('{TX_RAW}')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY msno, transaction_date ORDER BY membership_expire_date DESC
        ) = 1
    ) TO '{TX_OUT}' (FORMAT PARQUET)
""")
n_staged = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TX_OUT}')").fetchone()[0]
assert n_staged == n_groups, f"dedup mismatch: staged {n_staged} vs expected {n_groups}"
print(f"  {n_staged:,} rows written to {TX_OUT} (dedup rule applied and verified)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("STAGING COMPLETE (v1 files)")
for label, path in [("train_feb", TRAIN_OUT), ("transactions_v1", TX_OUT)]:
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {label:<18} {size_mb:>8.1f} MB  {path}")

print("\n  All row-count assertions passed. Phase 1 (v3) gate: OK to proceed to Phase 2 "
      "(dim_member, two cohorts).")