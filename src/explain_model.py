"""
explain_model.py  -  Phase 8 (v3): SHAP on the random forests

Question answered: which features push a user's churn probability up or down, and by how much?
  * Set A forest = the honest "drivers" view (history + static attributes, no current-membership snapshot).
  * Set B forest = same plus latest-transaction state, shown for comparison.
SHAP is computed on a random 10,000-user sample of the Mar (out-of-time) cohort.

What SHAP is and is not (interview): it explains what the MODEL uses, in probability units, additively -
base rate + sum of feature contributions = predicted probability.  It is not a causal claim: "auto-renew off
raises predicted churn" does not prove that switching auto-renew on would reduce churn.

Gates
  1. additivity: base value + sum(SHAP) reproduces predict_proba on 500 rows (max error < 0.002)
  2. one-hot / missing-indicator columns are folded back into their original feature before ranking
  3. soft check: the top-ranked driver should be among the strongest single features seen in the Phase 6
     tripwire (auto_renew_rate / last_is_auto_renew / n_tx); a surprise there is reported, not hidden

Needs : pip install shap
Reads : data/models/rf_A.joblib, rf_B.joblib, data/core/features_mar.parquet, feature_sets.json
Writes: data/core/shap_global.csv           (feature, mean_abs_shap, share, direction, in set)
        data/core/shap_dependence.csv       (feature, bin, mean_feature, mean_shap, n)  top-6 numeric features
        data/figures/shap_importance_A.png, shap_importance_B.png
"""
import json
import os
import re

import duckdb
import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    raise SystemExit("shap is not installed. Run:  pip install shap   then run this script again.")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

SAMPLE = 3_000
CHECK_ROWS = 500
SEED = 42
TOP_N = 15

os.makedirs(P("data", "figures"), exist_ok=True)
_con = duckdb.connect()


def read_pq(path):
    return _con.execute(f"SELECT * FROM read_parquet('{path.replace(chr(92), '/')}')").df()


with open(P("data", "core", "feature_sets.json")) as fh:
    fs = json.load(fh)
CAT = fs["categorical"]
SETS = {"A": fs["set_a"], "B": fs["set_b"]}

mar = read_pq(P("data", "core", "features_mar.parquet"))
for c in CAT:
    mar[c] = mar[c].astype("string").fillna("missing").astype(str)
sample = mar.sample(n=min(SAMPLE, len(mar)), random_state=SEED).reset_index(drop=True)
print(f"SHAP sample: {len(sample):,} Mar users, churn rate {100*sample['is_churn'].mean():.2f}%")


def original_feature(col_name, orig_cols):
    """Fold a transformed column name back to the feature it came from."""
    name = col_name.split("__", 1)[1] if "__" in col_name else col_name
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):]
    for c in sorted(orig_cols, key=len, reverse=True):      # longest first: 'last_payment_method_id' before 'city'
        if name == c or name.startswith(c + "_"):
            return c
    raise ValueError(f"cannot map transformed column '{name}' to an original feature")


def class1(sv):
    """shap returns a list per class (old) or an (n, features, classes) array (new)."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def base1(ev):
    b = np.asarray(ev.expected_value).ravel()
    return float(b[1] if len(b) > 1 else b[0])


results, dep_rows = [], []
for set_name, cols in SETS.items():
    print("\n" + "=" * 78 + f"\nSET {set_name}  ({len(cols)} features)\n" + "=" * 78)
    pipe = joblib.load(P("data", "models", f"rf_{set_name}.joblib"))
    pre, rf = pipe.named_steps["pre"], pipe.named_steps["model"]
    X = pre.transform(sample[cols])
    names = list(pre.get_feature_names_out())
    assert X.shape[1] == len(names)

    ex = shap.TreeExplainer(rf)
    sv = class1(ex.shap_values(X))
    assert sv.shape == X.shape, f"SHAP shape {sv.shape} vs X {X.shape}"

    # ---- gate 1: additivity against the model's own probabilities
    idx = np.arange(CHECK_ROWS)
    recon = base1(ex) + sv[idx].sum(axis=1)
    proba = rf.predict_proba(X[idx])[:, 1]
    err = np.abs(recon - proba).max()
    print(f"Additivity check ({CHECK_ROWS} rows): max |base + sum(SHAP) - predict_proba| = {err:.5f}")
    assert err < 0.002, f"SHAP is not additive (max error {err:.4f}) - do not use these values"
    print(f"Base value (average predicted churn): {base1(ex):.4f}")

    # ---- gate 2: fold back to original features
    owner = np.array([original_feature(n, cols) for n in names])
    folded = pd.DataFrame({f: sv[:, owner == f].sum(axis=1) for f in dict.fromkeys(owner)})
    mean_abs = folded.abs().mean().sort_values(ascending=False)
    total = mean_abs.sum()

    # direction for numeric features: correlation between the feature's value and its SHAP contribution
    direction = {}
    for f in mean_abs.index:
        if f in CAT:
            direction[f] = "categorical"
            continue
        x = pd.to_numeric(sample[f], errors="coerce")
        ok = x.notna() & (x.nunique() > 1)
        if ok.sum() < 100 or x[ok].nunique() < 2:
            direction[f] = "n/a"
            continue
        r = np.corrcoef(x[ok], folded.loc[ok, f])[0, 1]
        direction[f] = "higher value -> higher churn" if r > 0.1 else ("higher value -> lower churn" if r < -0.1 else "mixed / non-monotonic")

    for f, v in mean_abs.items():
        results.append({"set": set_name, "feature": f, "mean_abs_shap": v, "share": v / total, "direction": direction[f]})

    show = pd.DataFrame({"mean|SHAP|": mean_abs.round(4), "share %": (100 * mean_abs / total).round(1),
                         "direction": pd.Series(direction)}).head(TOP_N)
    print(f"\nTop {TOP_N} drivers of predicted churn probability (Set {set_name}, Mar sample):")
    print(show.to_string())
    top = mean_abs.index[0]
    if top not in ("auto_renew_rate", "last_is_auto_renew", "n_tx"):
        print(f"NOTE: top driver is '{top}', not one of the strongest single features from Phase 6 - worth a look.")

    # ---- dependence: how SHAP changes across the range of the top numeric features
    for f in [f for f in mean_abs.index if f not in CAT][:6]:
        x = pd.to_numeric(sample[f], errors="coerce")
        d = pd.DataFrame({"x": x, "s": folded[f]}).dropna()
        if d["x"].nunique() < 3:
            continue
        d["bin"] = pd.qcut(d["x"], q=min(10, d["x"].nunique()), duplicates="drop")
        g = d.groupby("bin", observed=True).agg(mean_feature=("x", "mean"), mean_shap=("s", "mean"), n=("s", "size"))
        for i, (_, r) in enumerate(g.iterrows()):
            dep_rows.append({"set": set_name, "feature": f, "bin": i + 1, "mean_feature": r["mean_feature"],
                             "mean_shap": r["mean_shap"], "n": int(r["n"])})

    # ---- chart (colors from the validated palette; one series so no legend needed)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
        t = (100 * mean_abs / total).head(TOP_N)[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=160, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.barh(t.index, t.values, color=BLUE, height=0.62)
        for yv, (name, v) in enumerate(t.items()):
            ax.text(v + 0.4, yv, f"{v:.1f}%", va="center", color=INK2, fontsize=9)
        ax.set_xlabel("Share of total mean |SHAP| (%)", color=INK2)
        ax.set_title(f"What drives predicted churn - random forest, Set {set_name}", loc="left", color=INK, fontsize=12.5)
        ax.tick_params(colors=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlim(0, t.max() * 1.15)
        fig.tight_layout()
        out = P("data", "figures", f"shap_importance_{set_name}.png")
        fig.savefig(out, facecolor=SURFACE)
        plt.close(fig)
        print(f"Wrote {out}")
    except ImportError:
        print("matplotlib not installed - skipped the chart")

pd.DataFrame(results).to_csv(P("data", "core", "shap_global.csv"), index=False)
pd.DataFrame(dep_rows).to_csv(P("data", "core", "shap_dependence.csv"), index=False)
print(f"\nWrote {P('data','core','shap_global.csv')} and shap_dependence.csv")
print("\nAll Phase 8 (v3) assertions passed.")