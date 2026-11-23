# SaaS Cohort Retention & Churn Driver Analysis — KKBox

Subscription churn on 2.4M music-streaming subscribers: a reproducible DuckDB pipeline, cohort retention and
survival analysis, and a churn model validated **out-of-time** on a later month it never saw in training.

**Dataset:** [WSDM Cup 2018 — KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
(Taiwan's largest music streaming service). `transactions.csv`: 21.5M billing events, 2.36M users,
2015-01-01 → 2017-02-28.

---

## Headline results

| | |
|---|---|
| **Out-of-time ROC-AUC** | **0.864** (random forest, full feature set, scored on a later month's cohort) |
| **Business lift** | The riskiest **10% of users contain 60% of all churners** — 6.0× lift over random targeting |
| **Strongest driver** | Auto-renew behaviour — 26% of total model attribution (SHAP) |
| **Retention split** | Users whose first payment was **paid** have a median lifetime of **410 days**; users who started on a **free trial**, **11 days** |

Published benchmarks on this dataset, same no-logs scope: logistic regression 0.854, XGBoost out-of-time 0.873.
This project lands inside that band using transactions and member attributes only.

---

## The result that matters most

AUC is not a business decision. The decile table is:

| Risk decile (Mar cohort) | Lift | Cumulative churners captured |
|---|---|---|
| Top 10% | 6.0× | 60.0% |
| Top 20% | — | 73.3% |

A retention team that can only call 10% of the base reaches **three-fifths of everyone who was going to leave**.

---

## What drives churn

From SHAP on the history-only model (no current-membership snapshot — see *Two feature sets* below):

| Driver group | Share of attribution | Direction |
|---|---|---|
| Auto-renew behaviour | ~28% | Rarely auto-renewing → higher churn |
| Amount of history (transactions, total paid, tenure) | ~18% | More history → lower churn |
| Cancellation history | ~17% | Any cancels → higher churn |
| Registration channel | 8% | Varies by channel |

**The story: churn here is billing mechanics, not disengagement.** Auto-renew state, cancellation history and
payment method dominate. A lapsed card or a switched-off renewal churns a subscriber who never decided to leave —
which is a far more actionable finding than "unengaged users churn", because the fix is operational.

Adding the user's membership snapshot at the scoring date (payment method, current plan, days until expiry) moves
AUC from 0.843 to 0.864 and promotes payment method to the #3 driver.

---

## Design decisions that make the result trustworthy

### 1. The trap in this dataset — and how it was caught

The competition ships `_v2` files described as containing data "until 3/31/2017". **They are a March-2017 delta
refresh, not the full history.** An early version of this project paired `train_v2` labels with `transactions_v2`
features and scored **0.868 AUC** — which looked like success and was pure leakage: `transactions_v2` is ~75%
March-2017 rows, and the label *is* the March renewal decision. The giveaway:

| Group | Users | Churn rate |
|---|---|---|
| Transactions only on/after 2017-03-01 | 840,892 (86.6%) | 4.9% |
| Has pre-March transactions | 92,686 (9.5%) | 17.8% |
| No transactions at all | 37,382 (3.9%) | **78.7%** |

"Does a March row exist" nearly *is* the label. The fix was to download the real v1 files and rebuild. That
retraction is documented rather than hidden, because catching your own leak is the skill being demonstrated.

### 2. Two cohorts, genuine out-of-time validation

| | Training cohort | Out-of-time test cohort |
|---|---|---|
| Labels | `train.csv` — 992,931 users, Feb-2017 expiries, 6.39% churn | `train_v2.csv` — 970,960 users, Mar-2017 expiries, 8.99% churn |
| Feature cut-off ("as-of date") | **2017-01-31** | **2017-02-28** |
| Feature source | `transactions.csv` | `transactions.csv` + pre-March rows of `transactions_v2.csv` |

A **fixed as-of date per cohort**, not a per-user cut-off. A per-user rule ("everything strictly before this user's
expiry") cannot exclude an *early* renewal — which is itself the label. A fixed date is uniform across both classes
and defensible in one sentence.

**Honest caveat:** 881,701 of the 970,960 March users (90.8%) also appear in the February cohort — they renewed in
February and come up for renewal again in March. So this is **out-of-time, not out-of-user**. It does not appear to
flatter the score: those users were non-churners in training, and AUC on March users *not* in February is **higher**
(0.867 vs 0.864 overall).

### 3. Two feature sets, both reported

- **Set A (25 features)** — static member attributes plus transaction *history* and ratios. No current-membership
  snapshot. **Out-of-time AUC 0.843.**
- **Set B (31 features)** — Set A plus state as of the scoring date: current plan, payment method, auto-renew flag,
  cancel flag, days until expiry. Legitimately knowable on the as-of date, but near-label in strength.
  **Out-of-time AUC 0.864.**

Published work shows that stripping near-label features drops this dataset from ~0.845 to ~0.721, so reporting only
the higher number would be misleading. Set A is quoted whenever the question is *what drives churn*.

### 4. Leakage controls, enforced in code

Every pipeline stage fails loudly rather than producing a quietly wrong number:

- The feature builder **asserts** the latest transaction date used equals the as-of date, and that no label-window
  column (`reconstructed_churn`, `last_expiry_full`, `window_visible`) reaches the feature table.
- A **single-feature AUC tripwire**: no individual feature may reach 0.90 alone. Strongest observed: 0.753.
- The trainer **halts** if any model exceeds 0.92 AUC — on this dataset that means a leak, not a breakthrough.
- Revenue is reconciled independently: the member-month table's total matches an independently computed figure to
  five decimal places.
- Retention and survival curves are asserted monotonic; SHAP values are asserted additive against the model's own
  probabilities.

### 5. Left truncation and censoring handled explicitly

Registration dates reach back to 2004, but transactions start 2015-01-01 — so early registrants have unobserved
tenure. Fixed **by design**: the survival clock starts at each user's *first transaction*, and the 2015-01 cohort
(548,792 users who were already subscribed before the file opens) is excluded. Right-censoring: a retention cell is
shown only when every user in that cohort is fully observable at that point, including the 30 days the churn rule
needs to resolve.

### 6. A data-quality finding that changed the analysis

Recent cohorts looked like collapsing retention — month-1 retention sliding 46% → 33% → 22%. It was a mix shift, not
a decline. Those cohorts are 70–86% free-trial signups, and trial users survive 30 days only 4–8% of the time.
**Paid-user retention over the same period was flat or improving** (60-day survival 78% → 86%). Retention is
therefore reported segmented by first-payment type; the pooled number is a Simpson's-paradox trap.

---

## Pipeline

```
data/raw/            train.csv, train_v2.csv, transactions.csv, members_v3.csv
   ↓  build_staging_v1.py        typed Parquet; dedup (msno, transaction_date) keeping max expiry
data/staging/
   ↓  build_dim_member_v3.py     one row per cohort user; churn label re-derived and reconciled
   ↓  build_fct_member_month.py  19.7M user-months; prepaid revenue spread across the days it covers
data/core/
   ↓  build_cohort_retention.py  survival table + retention triangle, segmented
   ↓  build_survival.py          Kaplan-Meier (computed in DuckDB, not a library)
   ↓  build_features.py          Set A / Set B, leakage-audited
   ↓  train_models.py            LR baseline → random forest; Feb holdout + Mar out-of-time
   ↓  explain_model.py           SHAP
   ↓  export_for_tableau.py      7 aggregate CSVs → data/tableau/
```

| Script | Phase | Gate it must pass |
|---|---|---|
| `check_v1_structure.py` | 0 | Signal exists before any pipeline is built (CV AUC 0.810) |
| `build_staging_v1.py` | 1 | Every staged row count reconciles with the raw file |
| `build_dim_member_v3.py` | 2 | Re-derived labels reproduce the Phase 0 agreement exactly (97.9% Feb) |
| `build_fct_member_month.py` | 3 | Revenue reconciles to 5 dp against `check_phase3_revenue.py` |
| `build_cohort_retention.py` | 4 | k=0 is 100%; retention never rises; segments partition the whole |
| `build_survival.py` | 5 | S(t) ∈ [0,1], never rises, events + censored = users |
| `build_features.py` | 6 | As-of date respected; no label-window columns; no feature ≥ 0.90 AUC |
| `train_models.py` | 7 | No model ≥ 0.92 AUC; forest must beat the LR baseline |
| `explain_model.py` | 8 | base + Σ SHAP reproduces `predict_proba` |

Each phase ends in a **gate** — a factual claim about the data that must hold before the next phase starts. If a
gate fails, the script stops. Two gates did fail during the build (negative survival durations; an unreconciled
revenue ratio) and both were investigated with a dedicated diagnostic script before any code was changed.

**Diagnostics** (read-only, never modify data): `inspect_negative_durations.py`,
`inspect_first_payment_by_cohort.py`, `inspect_day410_cliff.py`, `inspect_staging.py`, `inspect_label_window.py`.

---

## Full results

| Model | Feature set | Feb holdout AUC | **Mar out-of-time AUC** | Mar new-users AUC | Mar PR-AUC | Mar log loss |
|---|---|---|---|---|---|---|
| Logistic regression | A | 0.829 | 0.789 | 0.821 | 0.448 | 0.240 |
| Random forest | A | 0.879 | **0.843** | 0.858 | 0.545 | 0.219 |
| Logistic regression | B | 0.856 | 0.808 | 0.846 | 0.535 | 0.221 |
| **Random forest** | **B** | 0.893 | **0.864** | 0.867 | 0.592 | 0.206 |

Chance level: ROC-AUC 0.500; PR-AUC = base rate = 0.090 (Mar). A constant predictor scores ~0.30 log loss.

The logistic regression is not decoration — it is the benchmark the forest has to beat. It does, by ~0.05 AUC on
both feature sets.

**Calibration:** mean predicted churn in March is ~6.4% against an actual 8.99%, because the base rate shifts
between cohorts (6.39% → 8.99%). Probabilities are deliberately left unrecalibrated and the gap is reported;
ranking quality (AUC, lift) is unaffected.

---

## Survival analysis

Kaplan-Meier, computed directly in DuckDB — `S(t) = Π(1 − dᵢ/nᵢ)` is two aggregations and a window function, and
avoids a dependency on 2.4M rows.

| Still subscribed | 30 d | 60 d | 180 d | 365 d | Median |
|---|---|---|---|---|---|
| Paid first payment | 83.0% | 78.6% | 66.8% | 55.2% | **410 days** |
| Free first payment | 24.8% | 9.9% | 5.8% | 4.2% | **11 days** |
| All users | 59.4% | 50.7% | 42.0% | 34.4% | 61 days |

The pooled median (61 days) is an average of two populations that behave nothing alike — the split numbers are the
honest ones. The paid median sits on a sharp drop near day 410, so 365-day and 540-day survival are quoted in
preference to it.

---

## Scope decisions

**No listening logs.** `user_logs.csv` is 392M rows / ~30 GB. Published results on this dataset put
transactions-only logistic regression at 0.854 ROC-AUC against 0.865 with log features — ~0.01 AUC for 30 GB and
days of processing. Excluded deliberately, and the trade-off is quantified rather than assumed.
`user_logs_v2.csv` covers March 2017 only, which is after both cut-off dates; it is unusable by construction.

**DuckDB for all heavy lifting**, pandas only on aggregated output. The raw CSVs are never loaded into memory.

---

## Known limitations

1. **Out-of-time, not out-of-user** — 90.8% cohort overlap (quantified above; new-user AUC reported separately).
2. **First-churn definition** — a user who cancels, then returns months later and stays for years, is counted as
   churned at the first lapse. 83,969 users (3.6%) resubscribe after a gap.
3. **SHAP is attribution, not causation** — "auto-renew off raises predicted churn" is not "switching it on would
   retain them". Correlated features (transaction count, total paid, tenure) also split credit arbitrarily and are
   quoted as a group.
4. **Unexplained** — a sharp drop in the paid survival curve near day 410, and a 2015-03 cohort anomaly, are
   flagged as observations with no cause claimed. A diagnostic script exists for the first; it was not run.
5. **Probabilities are uncalibrated** across the base-rate shift (see above).

---

## Reproducing

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install duckdb pandas scikit-learn matplotlib shap joblib numpy

# place train.csv, train_v2.csv, transactions.csv, members_v3.csv in data/raw/
python src/check_v1_structure.py
python src/build_staging_v1.py
python src/build_dim_member_v3.py
python src/build_fct_member_month.py
python src/check_phase3_revenue.py
python src/build_cohort_retention.py
python src/build_survival.py
python src/build_features.py
python src/train_models.py            # --quick for a fast first pass
python src/explain_model.py
python src/export_for_tableau.py
```

Runtime on a consumer laptop: a few minutes per stage; the two random forests are the slowest at ~3 minutes each.
