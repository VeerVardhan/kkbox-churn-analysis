"""
export_for_tableau.py  -  Phase 9 (v3): aggregate the big tables down to dashboard-sized CSVs

Tableau Public cannot sensibly chew 20M-row member-month tables, and it should not have to: every number on the
dashboard is an aggregate.  This script does the aggregation in DuckDB and writes small, tidy CSVs - one per
dashboard sheet - into data/tableau/.  Each file is "tidy long": one row per thing being drawn.

Outputs (all in data/tableau/)
  1. cohort_retention.csv   segment x cohort_month x k         -> retention heatmap (3 segments)
  2. km_curves.csv          stratum x day                      -> survival curves (thinned to <= ~2,300 rows)
  3. revenue_monthly.csv    cohort x month x plan_tier         -> MRR / in-force members / ARPU over time
  4. model_results.csv      model x feature set                -> AUC / PR-AUC / log loss bar chart
  5. model_lift.csv         model x risk decile                -> lift + cumulative capture (the "who do we call
                                                                  first" chart - the most business-legible one)
  6. shap_global.csv        set x feature                      -> driver ranking
  7. shap_dependence.csv    set x feature x bin                -> how a driver's effect changes across its range

Reads : data/core/{cohort_retention,km_curves,fct_member_month_feb,fct_member_month_mar,predictions_mar}.parquet,
        data/core/{model_results.csv,shap_global.csv,shap_dependence.csv}
Writes: data/tableau/*.csv
"""
import os
import shutil

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")

OUT = P("data", "tableau")
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()


def pick(name):
    """The Phase 3 files may still carry the accidental double extension."""
    good, bad = P("data", "core", name + ".parquet"), P("data", "core", name + ".parquet.parquet")
    if os.path.exists(good):
        return good
    if os.path.exists(bad):
        return bad
    raise SystemExit(f"missing: {good}")


def dump(sql, out_name, note):
    path = os.path.join(OUT, out_name).replace("\\", "/")
    con.execute(f"COPY ({sql}) TO '{path}' (HEADER, DELIMITER ',')")
    n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}')").fetchone()[0]   # count the output, not a 2nd scan
    print(f"  {out_name:<26} {n:>7,} rows   {note}")
    return n


print("Writing dashboard extracts to data/tableau/ ...")

# ---------------------------------------------------------------- 1. retention heatmap
dump(f"""
    SELECT segment, CAST(cohort_month AS DATE) AS cohort_month, k, cohort_size, retained,
           ROUND(100 * retention_rate, 2) AS retention_pct
    FROM read_parquet('{pick('cohort_retention')}') ORDER BY segment, cohort_month, k
""", "cohort_retention.csv", "heatmap: rows = cohort, cols = k, colour = retention_pct")

# ---------------------------------------------------------------- 2. survival curves (thin to daily steps)
dump(f"""
    SELECT stratum, t AS days_since_first_payment, n_at_risk, d AS churn_events,
           ROUND(100 * survival, 3) AS survival_pct
    FROM read_parquet('{pick('km_curves')}')
    WHERE t <= 760 ORDER BY stratum, t
""", "km_curves.csv", "line chart: x = days, y = survival_pct, colour = stratum")

# ---------------------------------------------------------------- 3. revenue / members by month
fct_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pick('fct_member_month_feb')}')").fetchall()}
need = {"msno", "month_start", "normalized_revenue", "plan_tier", "membership_in_force_eom"}
missing = need - fct_cols
assert not missing, f"fct_member_month is missing expected columns: {missing}"
auto = "is_auto_renew" if "is_auto_renew" in fct_cols else "NULL"

parts = " UNION ALL ".join(
    f"SELECT '{lbl}' AS cohort, * FROM read_parquet('{pick('fct_member_month_' + sfx)}')"
    for lbl, sfx in (("Feb", "feb"), ("Mar", "mar")))
dump(f"""
    WITH f AS ({parts})
    SELECT cohort, CAST(month_start AS DATE) AS month_start,
           COALESCE(plan_tier, 'no_active_plan') AS plan_tier,
           COUNT(*)                                             AS user_months,
           SUM(COALESCE(membership_in_force_eom, FALSE)::INT)   AS members_in_force,
           ROUND(SUM(normalized_revenue), 0)                    AS normalized_revenue,
           ROUND(AVG({auto}::DOUBLE), 3)                        AS auto_renew_share,
           ROUND(SUM(normalized_revenue) /
                 NULLIF(SUM(COALESCE(membership_in_force_eom, FALSE)::INT), 0), 2) AS arpu_in_force
    FROM f GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
""", "revenue_monthly.csv", "area/line: x = month, y = normalized_revenue or members_in_force, colour = plan_tier")

# ---------------------------------------------------------------- 4. model scoreboard
res = P("data", "core", "model_results.csv")
dump(f"""
    SELECT model, "set" AS feature_set, n_features,
           ROUND(feb_auc, 4) AS feb_holdout_auc,
           ROUND(mar_auc, 4) AS mar_out_of_time_auc,
           ROUND(mar_auc_new_users_only, 4) AS mar_new_users_auc,
           ROUND(mar_pr_auc, 4) AS mar_pr_auc,
           ROUND(mar_logloss, 4) AS mar_log_loss
    FROM read_csv_auto('{res}') ORDER BY mar_auc DESC
""", "model_results.csv", "bar chart: AUC by model x feature set")

# ---------------------------------------------------------------- 5. lift / decile capture
pred_path = pick("predictions_mar")
pred_cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pred_path}')").fetchall()]
model_cols = [c for c in pred_cols if c.startswith("p_")]
assert model_cols, f"no p_* prediction columns in {pred_path}"
union = " UNION ALL ".join(
    f"SELECT '{c[2:]}' AS model, is_churn, {c} AS p FROM read_parquet('{pred_path}')" for c in model_cols)
dump(f"""
    WITH long AS ({union}),
    ranked AS (SELECT model, is_churn, p, NTILE(10) OVER (PARTITION BY model ORDER BY p DESC) AS risk_decile FROM long),
    agg AS (
        SELECT model, risk_decile, COUNT(*) AS users, SUM(is_churn) AS churners,
               AVG(is_churn::DOUBLE) AS actual_rate, AVG(p) AS mean_predicted
        FROM ranked GROUP BY 1, 2)
    SELECT model, risk_decile, users, churners,
           ROUND(100 * actual_rate, 2)   AS actual_churn_pct,
           ROUND(100 * mean_predicted, 2) AS predicted_churn_pct,
           ROUND(actual_rate / (SUM(churners) OVER (PARTITION BY model)
                                / SUM(users) OVER (PARTITION BY model)), 2) AS lift,
           ROUND(100.0 * SUM(churners) OVER (PARTITION BY model ORDER BY risk_decile)
                 / SUM(churners) OVER (PARTITION BY model), 1) AS cumulative_churners_captured_pct
    FROM agg ORDER BY model, risk_decile
""", "model_lift.csv", "bar: x = risk_decile (1 = riskiest), y = lift; line = cumulative capture")

# ---------------------------------------------------------------- 6-7. SHAP (already small CSVs)
for src, dst in (("shap_global.csv", "shap_global.csv"), ("shap_dependence.csv", "shap_dependence.csv")):
    s = P("data", "core", src)
    if os.path.exists(s):
        shutil.copyfile(s, os.path.join(OUT, dst))
        n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{s}')").fetchone()[0]
        print(f"  {dst:<26} {n:>7,} rows   copied from data/core")
    else:
        print(f"  {dst:<26}     ---   not found (run explain_model.py first)")

# ---------------------------------------------------------------- sanity read-out
print("\nSanity figures to check against the dashboard once it is built:")
print(con.execute(f"""
    SELECT model, MAX(CASE WHEN risk_decile = 1 THEN lift END) AS top_decile_lift,
           MAX(CASE WHEN risk_decile = 1 THEN cumulative_churners_captured_pct END) AS captured_in_top_10pct,
           MAX(CASE WHEN risk_decile = 2 THEN cumulative_churners_captured_pct END) AS captured_in_top_20pct
    FROM read_csv_auto('{os.path.join(OUT, 'model_lift.csv').replace(chr(92), '/')}')
    GROUP BY 1 ORDER BY captured_in_top_10pct DESC
""").df().to_string(index=False))

print(f"\nAll extracts written to {OUT}")
print("Phase 9 (v3) export step done - next: build the workbook from these 7 CSVs.")