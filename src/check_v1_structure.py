"""
src/check_v1_structure.py

Phase 0 RE-VERIFICATION (v3) - run after discovering the _v2 files are a March refresh,
not the full dataset. This checks the REAL history files: train.csv + transactions.csv.

Four checks, in order:
  1. Structure    - row counts and date range of transactions.csv (must be ~21.5M rows,
                    2015-01-01 to 2017-02-28). Confirms the download is the right file,
                    not another copy of the _v2 refresh.
  2. Join coverage - what % of train.csv (Feb cohort) users appear in transactions.csv
                     and members_v3.csv.
  3. Label reconstruction - re-derive is_churn for the Feb cohort using the official
                     30-day rule, from transactions.csv alone. Because the Feb cohort's
                     outcome window (Feb 2017) closes INSIDE transactions.csv (which runs
                     to 2017-02-28), this should be checkable for most of the cohort -
                     unlike the old v2 check, which could only check 377 users.
  4. Signal check  - a handful of features, each computed strictly on or before the
                     as-of date (2017-01-31), fed into a 5-fold CV logistic regression.
                     This is the real screen for whether the project has signal at all.

Run from anywhere - paths are resolved relative to this file's location.
"""

import os
import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")

TRAIN_FEB = os.path.join(RAW, "train.csv").replace("\\", "/")
TRANSACTIONS_V1 = os.path.join(RAW, "transactions.csv").replace("\\", "/")
MEMBERS = os.path.join(RAW, "members_v3.csv").replace("\\", "/")

AS_OF = "2017-01-31"  # Decision 6 (v3): fixed as-of date for the Feb cohort

for name, path in [("train.csv", TRAIN_FEB), ("transactions.csv", TRANSACTIONS_V1),
                    ("members_v3.csv", MEMBERS)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path} - check data/raw/")

con = duckdb.connect()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


con.execute(f"""
    CREATE OR REPLACE VIEW tx AS
    SELECT *,
        strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS tx_date,
        strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS expire_date
    FROM read_csv_auto('{TRANSACTIONS_V1}')
""")

# ---------------------------------------------------------------------------
# CHECK 1 - structure: is this really the v1 file, not another _v2 copy?
# ---------------------------------------------------------------------------
section("CHECK 1: transactions.csv structure (confirming this is the real history)")

struct = con.execute("""
    SELECT COUNT(*) AS rows, COUNT(DISTINCT msno) AS distinct_users,
           MIN(tx_date) AS min_date, MAX(tx_date) AS max_date
    FROM tx
""").fetchdf()
print(struct.to_string(index=False))

n_rows = struct["rows"][0]
max_date = struct["max_date"][0]
if n_rows < 15_000_000:
    print(f"\n  *** WARNING: only {n_rows:,} rows. Published counts for the real "
          f"transactions.csv are ~21.5M. This may still be a _v2-style file. STOP and check "
          f"the download before proceeding. ***")
else:
    print(f"\n  {n_rows:,} rows - consistent with the real transaction history (expect ~21.5M).")

if str(max_date) > "2017-02-28":
    print(f"  *** WARNING: max date {max_date} is after 2017-02-28. transactions.csv should "
          f"stop at end of February 2017. This looks like the _v2 file renamed. STOP. ***")

# ---------------------------------------------------------------------------
# CHECK 2 - join coverage of the Feb cohort
# ---------------------------------------------------------------------------
section("CHECK 2: join coverage of train.csv (Feb cohort) into transactions / members")

coverage = con.execute(f"""
    WITH t   AS (SELECT DISTINCT msno FROM read_csv_auto('{TRAIN_FEB}')),
         txd AS (SELECT DISTINCT msno FROM tx),
         mb  AS (SELECT DISTINCT msno FROM read_csv_auto('{MEMBERS}'))
    SELECT
        COUNT(*)        AS train_users,
        COUNT(txd.msno) AS in_transactions,
        COUNT(mb.msno)  AS in_members
    FROM t
    LEFT JOIN txd ON txd.msno = t.msno
    LEFT JOIN mb  ON mb.msno  = t.msno
""").fetchdf()
print(coverage.to_string(index=False))
n = coverage["train_users"][0]
for col in ["in_transactions", "in_members"]:
    print(f"  {col}: {100 * coverage[col][0] / n:.1f}% coverage")

churn_rate = con.execute(f"""
    SELECT AVG(is_churn) AS churn_rate, COUNT(*) AS n FROM read_csv_auto('{TRAIN_FEB}')
""").fetchdf()
print(f"\n  train.csv (Feb cohort) churn rate: {churn_rate['churn_rate'][0]*100:.2f}% "
      f"({churn_rate['n'][0]} users) - published figure is 6.39%")

# ---------------------------------------------------------------------------
# CHECK 3 - label reconstruction for the Feb cohort
# ---------------------------------------------------------------------------
section("CHECK 3: re-derive is_churn for the Feb cohort from transactions.csv")

recon = con.execute(f"""
    WITH last_expire AS (
        SELECT msno, expire_date AS last_expire_date
        FROM tx
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY msno ORDER BY tx_date DESC, expire_date DESC
        ) = 1
    ),
    visible AS (
        -- 30-day renewal window must be fully inside transactions.csv's own max date
        SELECT msno, last_expire_date
        FROM last_expire
        WHERE last_expire_date + INTERVAL 30 DAY <= DATE '{max_date}'
    ),
    renewed AS (
        SELECT DISTINCT v.msno
        FROM visible v
        JOIN tx t
          ON t.msno = v.msno
         AND t.tx_date > v.last_expire_date
         AND t.tx_date <= v.last_expire_date + INTERVAL 30 DAY
    ),
    reconstructed AS (
        SELECT v.msno,
               CASE WHEN r.msno IS NULL THEN 1 ELSE 0 END AS reconstructed_churn
        FROM visible v
        LEFT JOIN renewed r ON r.msno = v.msno
    )
    SELECT tr.is_churn AS supplied_churn, rc.reconstructed_churn, COUNT(*) AS n
    FROM reconstructed rc
    JOIN read_csv_auto('{TRAIN_FEB}') tr ON tr.msno = rc.msno
    GROUP BY 1, 2
    ORDER BY 1, 2
""").fetchdf()
print(recon.to_string(index=False))

total_checked = recon["n"].sum()
agree = recon[recon["supplied_churn"] == recon["reconstructed_churn"]]["n"].sum()
print(f"\n  Agreement on the visible (non-censored) subset: "
      f"{agree}/{total_checked} = {100*agree/total_checked:.1f}%")

visible_count = con.execute(f"""
    WITH last_expire AS (
        SELECT msno, expire_date AS last_expire_date
        FROM tx
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY msno ORDER BY tx_date DESC, expire_date DESC
        ) = 1
    )
    SELECT
        COUNT(*) AS total_users_with_transactions,
        SUM(CASE WHEN last_expire_date + INTERVAL 30 DAY <= DATE '{max_date}'
                 THEN 1 ELSE 0 END) AS visible_for_reconstruction
    FROM last_expire
""").fetchdf()
print(f"  Users with transactions: {visible_count['total_users_with_transactions'][0]}, "
      f"of which {visible_count['visible_for_reconstruction'][0]} have a fully-visible "
      f"30-day window. Expect this to cover the MAJORITY of the Feb cohort now - "
      f"unlike the old v2 check (377 users), this file's outcome window closes inside it.")

# ---------------------------------------------------------------------------
# CHECK 4 - signal check, features strictly on/before the as-of date
# ---------------------------------------------------------------------------
section(f"CHECK 4: signal check - features as of {AS_OF}, 5-fold CV logistic regression AUC")

features = con.execute(f"""
    WITH tx_agg AS (
        SELECT
            msno,
            COUNT(*)                                        AS n_transactions,
            AVG(payment_plan_days)                           AS avg_plan_days,
            AVG(actual_amount_paid)                          AS avg_amount_paid,
            AVG(plan_list_price - actual_amount_paid)        AS avg_discount,
            AVG(CAST(is_cancel AS DOUBLE))                   AS cancel_rate,
            AVG(CAST(is_auto_renew AS DOUBLE))               AS auto_renew_rate,
            DATE_DIFF('day', MAX(tx_date), DATE '{AS_OF}')   AS days_since_last_tx
        FROM tx
        WHERE tx_date <= DATE '{AS_OF}'
        GROUP BY msno
    )
    SELECT
        tr.msno, tr.is_churn,
        tx.n_transactions, tx.avg_plan_days, tx.avg_amount_paid, tx.avg_discount,
        tx.cancel_rate, tx.auto_renew_rate, tx.days_since_last_tx,
        mb.city, mb.bd, mb.gender, mb.registered_via
    FROM read_csv_auto('{TRAIN_FEB}') tr
    LEFT JOIN tx_agg tx ON tx.msno = tr.msno
    LEFT JOIN read_csv_auto('{MEMBERS}') mb ON mb.msno = tr.msno
""").fetchdf()

print(f"  Feature table: {features.shape[0]} rows, {features.shape[1]} columns")
print(f"  Missing values per column:\n{features.isna().sum().to_string()}")

y = features["is_churn"]
num_cols = ["n_transactions", "avg_plan_days", "avg_amount_paid", "avg_discount",
            "cancel_rate", "auto_renew_rate", "days_since_last_tx", "bd"]
X = features[num_cols].copy()

gender_dummies = pd.get_dummies(features["gender"], prefix="gender", dummy_na=True)
X = pd.concat([X, gender_dummies], axis=1)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(f"\n  5-fold CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"  Individual folds: {np.round(scores, 3)}")
print(f"  Reference band (published, no-logs scope): 0.72-0.87. Near 0.5 = no signal, stop.")
print(f"  Above ~0.92 = still leaking somehow - check that tx_date <= {AS_OF} filter above.")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
section("GATE VERDICT")
agree_pct = 100 * agree / total_checked
print(f"  Check 1 (structure): {n_rows:,} rows, max date {max_date} - "
      f"{'OK' if n_rows >= 15_000_000 and str(max_date) <= '2017-02-28' else 'FAILED - see warnings above'}")
print(f"  Check 2 (coverage): see table above")
print(f"  Check 3 (label reconstruction): {agree_pct:.1f}% agreement on "
      f"{total_checked:,} users with a visible window "
      f"(of {visible_count['total_users_with_transactions'][0]:,} users with transactions)")
print(f"  Check 4 (signal, as-of {AS_OF}): {scores.mean():.3f} +/- {scores.std():.3f}")
print("\n  All four must hold before Phase 1: structure OK, coverage not gutted, "
      "label agreement >= 90%, AUC between ~0.65 and ~0.92.")
print("  If this passes, proceed to Phase 1 (extend build_staging.py for train.csv / "
      "transactions.csv). Do not build dim_member or features until this report is reviewed.")