"""
build_survival.py  -  Phase 5 (v3): Kaplan-Meier survival, stratified by first-payment type

Why DuckDB and not lifelines: 2.36M users.  A KM estimate only needs, per distinct duration t, the number of
events d_t and the number still at risk n_t - two GROUP BYs and a window function.  No new dependency, and it
is easy to explain:  S(t) = product over event times t_i <= t of (1 - d_i / n_i).

Left truncation: handled by DESIGN, not by lifelines' entry=.  Every user's clock starts at their own first
transaction and the 2015-01 cohort (who were already subscribed before the file starts) is excluded, so every
remaining user is observed from clock-time 0.  (Say this in an interview: "I fixed left truncation by choosing the
clock origin and dropping the truncated cohort; entry= would be the alternative if I wanted to keep them.")

Right censoring: users with no knowable churn are censored at 2017-02-28 (duration = days since first payment).
A censored user at time t is counted as at risk AT t (standard convention) and leaves afterwards.

Reads : data/core/user_survival.parquet
Writes: data/core/km_curves.parquet           (stratum, t, n_at_risk, d, survival)
        data/figures/km_by_first_payment.png  (only if matplotlib is installed)
"""
import os
import numpy as np
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")

os.makedirs(P("data", "core"), exist_ok=True)
os.makedirs(P("data", "figures"), exist_ok=True)

con = duckdb.connect()
con.execute(f"CREATE VIEW us AS SELECT * FROM read_parquet('{P('data','core','user_survival.parquet')}')")

con.execute("""
    CREATE OR REPLACE TEMP TABLE base AS
    SELECT 'all' AS stratum, duration_days AS t, event::INT AS e FROM us WHERE NOT is_left_truncated_cohort
    UNION ALL
    SELECT first_payment_type, duration_days, event::INT FROM us WHERE NOT is_left_truncated_cohort
""")

con.execute("""
    CREATE OR REPLACE TEMP TABLE km AS
    WITH agg AS (
        SELECT stratum, t, SUM(e) AS d, COUNT(*) AS n_exit FROM base GROUP BY 1, 2
    ),
    risk AS (
        SELECT stratum, t, d, n_exit,
               SUM(n_exit) OVER (PARTITION BY stratum) -
               COALESCE(SUM(n_exit) OVER (PARTITION BY stratum ORDER BY t
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS n_at_risk
        FROM agg
    )
    SELECT stratum, t, n_at_risk, d, n_exit - d AS censored,
           CASE WHEN SUM((d >= n_at_risk)::INT) OVER w > 0 THEN 0.0
                ELSE EXP(SUM(CASE WHEN d < n_at_risk THEN LN(1.0 - d::DOUBLE / n_at_risk) ELSE 0 END) OVER w)
           END AS survival
    FROM risk
    WINDOW w AS (PARTITION BY stratum ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
""")

# ------------------------------------------------------------------ gates
bad_range = con.execute("SELECT COUNT(*) FROM km WHERE survival < 0 OR survival > 1").fetchone()[0]
assert bad_range == 0, f"{bad_range} KM values outside [0, 1]"

rises = con.execute("""
    SELECT COUNT(*) FROM (SELECT survival, LAG(survival) OVER (PARTITION BY stratum ORDER BY t) AS prev FROM km)
    WHERE prev IS NOT NULL AND survival > prev + 1e-12""").fetchone()[0]
assert rises == 0, f"{rises} points where the KM curve rises"

# the event / censor counts must reconcile with the user table, and the strata must partition 'all'
chk = con.execute("""
    SELECT stratum, SUM(d) AS events, SUM(censored) AS censored, SUM(d + censored) AS users,
           MAX(n_at_risk) AS n0
    FROM km GROUP BY 1 ORDER BY 1""").df()
print("\nReconciliation (events + censored must equal users; n_at_risk at the first time must equal users):")
print(chk.to_string(index=False))
assert (chk["users"] == chk["n0"]).all(), "n at risk at t_min differs from users - risk set is wrong"
for stratum, where in (("all", "TRUE"), ("paid_first", "first_payment_type='paid_first'"),
                       ("free_first", "first_payment_type='free_first'")):
    u, ev = con.execute(f"SELECT COUNT(*), SUM(event::INT) FROM us WHERE NOT is_left_truncated_cohort AND {where}").fetchone()
    row = chk[chk["stratum"] == stratum].iloc[0]
    assert int(row["users"]) == u and int(row["events"]) == ev, f"{stratum}: KM counts {row['users']}/{row['events']} vs user table {u}/{ev}"
print("KM counts match the user table for all three strata.")

con.execute(f"COPY (SELECT stratum, t, n_at_risk, d, survival FROM km ORDER BY 1, 2) "
            f"TO '{P('data','core','km_curves.parquet')}' (FORMAT PARQUET)")
print(f"Wrote {P('data','core','km_curves.parquet')}")

# ------------------------------------------------------------------ readable summary
df = con.execute("SELECT stratum, t, survival FROM km ORDER BY stratum, t").df()
horizons = [30, 60, 90, 180, 365, 540]


def surv_at(g, h):
    """S(h): last KM value at or before h (curve is a step function)."""
    idx = np.searchsorted(g["t"].to_numpy(), h, side="right") - 1
    return g["survival"].to_numpy()[idx] if idx >= 0 else 1.0


rows = []
for stratum in ("all", "paid_first", "free_first"):
    g = df[df["stratum"] == stratum].reset_index(drop=True)
    below = g[g["survival"] <= 0.5]
    median = f"{int(below['t'].iloc[0])} d" if len(below) else "not reached"
    rows.append({"stratum": stratum, **{f"S({h}d) %": round(100 * surv_at(g, h), 1) for h in horizons},
                 "median": median})
import pandas as pd
print("\nKaplan-Meier survival (percent still subscribed, by days since first payment):")
print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: 'survival' here is 'has not yet had a >30-day lapse' - KKBox's own churn rule, first event only.")

# Tie convention: KM's S(t) = P(T > t).  The Phase 4 heatmap's "retained at month k" = duration >= 30k = P(T >= 30k)
# = S(30k - 1).  Lots of plans end exactly on day 30 / 60, so the two differ by many points at those days.
rows2 = []
for stratum in ("all", "paid_first", "free_first"):
    g = df[df["stratum"] == stratum].reset_index(drop=True)
    rows2.append({"stratum": stratum, **{f"month {k} %": round(100 * surv_at(g, 30 * k - 1), 1) for k in (1, 2, 3, 6, 12)}})
print("\nSame curve in the heatmap's definition ('retained at month k' = still covered on day 30k = S(30k - 1)):")
print(pd.DataFrame(rows2).to_string(index=False))

# ------------------------------------------------------------------ chart (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8d8c86"   # validated slots 1-2 + neutral for 'all'
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for stratum, color, ls, label in (("paid_first", BLUE, "-", "Paid first payment"),
                                      ("free_first", ORANGE, "-", "Free first payment (trial/promo)"),
                                      ("all", GRAY, (0, (4, 3)), "All users")):
        g = df[df["stratum"] == stratum]
        ax.step(g["t"], 100 * g["survival"], where="post", color=color, lw=2, ls=ls, label=label)
    for x in (30, 60, 365):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Days since first payment", color=INK2)
    ax.set_ylabel("Still subscribed (%)", color=INK2)
    ax.set_title("Survival by type of first payment", loc="left", color=INK, fontsize=12.5)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    out = P("data", "figures", "km_by_first_payment.png")
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}")
except ImportError:
    print("matplotlib not installed - skipped the chart (the km_curves.parquet table is the real output).")

print("\nAll Phase 5 (v3) assertions passed.")