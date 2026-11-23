"""
train_models.py  -  Phase 7 (v3): logistic-regression baseline, then random forest; Set A and Set B; out-of-time on March

Design
  * TRAIN on 80% of the Feb cohort (as-of 2017-01-31).
  * TEST 1 (in-time):    the other 20% of Feb.
  * TEST 2 (out-of-time): the whole Mar cohort (as-of 2017-02-28) - a different month, different labels, never
                          seen in training.  This is the headline number.
  * 4 models:  {logistic regression, random forest} x {Set A, Set B}.   Logistic regression is the baseline: a forest
    that does not beat it means something is wrong.
  * Metrics: ROC-AUC (ranking), PR-AUC (average precision; baseline = churn rate), log loss (probability quality).
  * Honest band for this dataset without logs: roughly 0.72-0.87 ROC-AUC.  Anything above 0.92 stops the script.

Extra rigour
  * Users who appear in BOTH cohorts could let the model "recognise" a user.  Overlap is counted and every
    out-of-time metric is also reported on Mar users who are NOT in Feb.
  * Mar's churn rate (8.99%) is higher than Feb's (6.39%), so mean predicted vs actual rate is printed -
    a gap there is base-rate shift, not a bug, and explains why log loss gets worse out of time.

Run   python src/train_models.py --quick     first look: trains on 25% of the Feb training rows (fast)
      python src/train_models.py             full run

Reads : data/core/features_feb.parquet, features_mar.parquet, feature_sets.json
Writes: data/core/model_results.csv, data/core/predictions_mar.parquet, data/models/{lr,rf}_{A,B}.joblib
"""
import argparse
import json
import os
import time

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

LEAK_AUC = 0.92
SEED = 42


_con = duckdb.connect()


def read_pq(path):
    """Parquet via DuckDB (no pyarrow dependency, same engine as every earlier phase)."""
    return _con.execute(f"SELECT * FROM read_parquet('{path.replace(chr(92), '/')}')").df()


def write_pq(df, path):
    _con.register("df_out", df)
    _con.execute(f"COPY df_out TO '{path.replace(chr(92), '/')}' (FORMAT PARQUET)")
    _con.unregister("df_out")


ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="train on 25%% of the Feb training rows")
args = ap.parse_args()

os.makedirs(P("data", "models"), exist_ok=True)
with open(P("data", "core", "feature_sets.json")) as fh:
    fs = json.load(fh)
SETS = {"A": fs["set_a"], "B": fs["set_b"]}
CAT = fs["categorical"]

feb = read_pq(P("data", "core", "features_feb.parquet"))
mar = read_pq(P("data", "core", "features_mar.parquet"))
forbidden = {"last_expiry_full", "window_visible", "is_window_censored", "reconstructed_churn", "is_churn_supplied"}
for name, df in (("feb", feb), ("mar", mar)):
    assert not (forbidden & set(SETS["B"])), "forbidden column in the declared feature set"
    for c in CAT:                       # categoricals as clean strings so 'missing' is its own level
        df[c] = df[c].astype("string").fillna("missing").astype(str)
    print(f"{name}: {len(df):,} users, churn rate {100*df['is_churn'].mean():.2f}%")

overlap = set(feb["msno"]) & set(mar["msno"])
print(f"Users present in BOTH cohorts: {len(overlap):,} ({100*len(overlap)/len(mar):.1f}% of Mar)")
mar_new = ~mar["msno"].isin(overlap).to_numpy()

feb_tr, feb_te = train_test_split(feb, test_size=0.2, random_state=SEED, stratify=feb["is_churn"])
if args.quick:
    feb_tr = feb_tr.sample(frac=0.25, random_state=SEED)
print(f"Train rows: {len(feb_tr):,}{'  (--quick)' if args.quick else ''}   Feb holdout: {len(feb_te):,}   Mar (out-of-time): {len(mar):,}")


def make_pipeline(kind, cols):
    cat = [c for c in cols if c in CAT]
    num = [c for c in cols if c not in CAT]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "lr":
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2000, sparse_output=False), cat),
    ])
    if kind == "lr":
        model = LogisticRegression(max_iter=500, C=1.0)
    else:
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=100, max_features="sqrt",
                                       n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("model", model)])


def metrics(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


rows, preds = [], pd.DataFrame({"msno": mar["msno"], "is_churn": mar["is_churn"]})
for set_name, cols in SETS.items():
    for kind, label in (("lr", "Logistic regression"), ("rf", "Random forest")):
        t0 = time.time()
        pipe = make_pipeline(kind, cols)
        pipe.fit(feb_tr[cols], feb_tr["is_churn"])
        p_in = pipe.predict_proba(feb_te[cols])[:, 1]
        p_oot = pipe.predict_proba(mar[cols])[:, 1]
        m_in, m_oot = metrics(feb_te["is_churn"], p_in), metrics(mar["is_churn"], p_oot)
        m_new = metrics(mar["is_churn"][mar_new], p_oot[mar_new])
        rows.append({"model": label, "set": set_name, "n_features": len(cols),
                     "feb_auc": m_in["roc_auc"], "feb_pr_auc": m_in["pr_auc"], "feb_logloss": m_in["log_loss"],
                     "mar_auc": m_oot["roc_auc"], "mar_pr_auc": m_oot["pr_auc"], "mar_logloss": m_oot["log_loss"],
                     "mar_auc_new_users_only": m_new["roc_auc"],
                     "mar_mean_pred": p_oot.mean(), "mar_actual_rate": mar["is_churn"].mean(),
                     "seconds": time.time() - t0})
        preds[f"p_{kind}_{set_name}"] = p_oot
        joblib.dump(pipe, P("data", "models", f"{kind}_{set_name}.joblib"))
        print(f"  {label:<20} Set {set_name}:  Feb AUC {m_in['roc_auc']:.3f}   Mar AUC {m_oot['roc_auc']:.3f}   "
              f"({time.time()-t0:.0f}s)")
        assert max(m_in["roc_auc"], m_oot["roc_auc"]) < LEAK_AUC, \
            f"AUC {max(m_in['roc_auc'], m_oot['roc_auc']):.3f} >= {LEAK_AUC}: almost certainly a leak - stop and audit"

res = pd.DataFrame(rows)
res.to_csv(P("data", "core", "model_results.csv"), index=False)
write_pq(preds, P("data", "core", "predictions_mar.parquet"))

pd.set_option("display.width", 200)
print("\n" + "=" * 100 + "\nRESULTS  (train: 80% of Feb | Feb = in-time holdout | Mar = out-of-time)\n" + "=" * 100)
show = res[["model", "set", "n_features", "feb_auc", "mar_auc", "mar_auc_new_users_only", "mar_pr_auc", "mar_logloss"]]
print(show.round(3).to_string(index=False))
print(f"\nChance level: ROC-AUC 0.500; PR-AUC = churn rate = {mar['is_churn'].mean():.3f} (Mar), "
      f"{feb['is_churn'].mean():.3f} (Feb).")
print("\nBase-rate shift (Mar): mean predicted probability vs actual churn rate")
print(res[["model", "set", "mar_mean_pred", "mar_actual_rate"]].round(4).to_string(index=False))
print("\nHonest band for this dataset without logs: ~0.72-0.87 ROC-AUC. Published: transactions-only LR 0.854, "
      "XGBoost March out-of-time 0.873.")
print("\nAll Phase 7 (v3) assertions passed.")