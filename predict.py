"""Predict the 2026 Qatar Grand Prix finishing order.

Usage:
    python predict.py                      # pre-qualifying model (form features only)
    python predict.py --grid grid.txt      # post-qualifying: grid.txt lists driverIds in grid order, one per line
    python predict.py --backtest           # walk-forward evaluation on 2025-2026 races
    python predict.py --quali              # Saturday qualifying order (pole / top-3 probabilities)

Model: ridge regression predicts finishing position from rolling driver/team form,
then a Monte Carlo simulation over historical prediction errors turns point predictions
into win / podium / points probabilities.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
TARGET_SEASON, TARGET_ROUND, TARGET_CIRCUIT = 2026, 22, "losail"
N_SIMS = 20_000
FORM_FEATURES = [
    "drv_form5", "drv_quali5", "team_form5", "team_quali5",
    "drv_pts_rate", "team_pts_rate", "drv_dnf10", "drv_circuit_avg",
    "new_regs", "season_progress",
]
GRID_FEATURES = FORM_FEATURES + ["quali_pos"]


def load():
    res = pd.read_csv(DATA / "results.csv")
    q = pd.read_csv(DATA / "qualifying.csv")
    df = res.merge(q, on=["season", "round", "driver"], how="left")
    df["dnf"] = ~df.status.str.match(r"^(Finished|\+\d+ Laps?|Lapped)$")
    df["race_idx"] = df.season * 100 + df["round"]
    return df.sort_values(["race_idx", "position"]).reset_index(drop=True)


def append_target_race(df, grid=None):
    """Add placeholder rows for the target race using the latest race's entry list."""
    last = df[df.race_idx == df.race_idx.max()]
    target = last[["driver", "driver_name", "team", "team_name"]].copy()
    target["season"], target["round"] = TARGET_SEASON, TARGET_ROUND
    target["circuit"], target["race"] = TARGET_CIRCUIT, "Qatar Grand Prix"
    target["race_idx"] = TARGET_SEASON * 100 + TARGET_ROUND
    target["quali_pos"] = np.nan
    if grid:
        order = {d: i + 1 for i, d in enumerate(grid)}
        unknown = set(grid) - set(target.driver)
        if unknown:
            raise SystemExit(f"Unknown driverIds in grid: {sorted(unknown)}. Valid: {sorted(target.driver)}")
        target["quali_pos"] = target.driver.map(order)
    return pd.concat([df, target], ignore_index=True)


def build_features(df):
    """Every feature uses only races strictly before the row's race (shift(1) before rolling)."""
    df = df.sort_values(["race_idx", "driver"]).reset_index(drop=True)
    g = df.groupby("driver")
    df["drv_form5"] = g.position.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["drv_quali5"] = g.quali_pos.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["drv_dnf10"] = g.dnf.transform(lambda s: s.astype(float).shift(1).rolling(10, min_periods=1).mean())

    # Points per race within the current season only, since 2026 reset the competitive order
    gs = df.groupby(["driver", "season"])
    df["drv_pts_rate"] = gs.points.transform(lambda s: s.shift(1).expanding().mean())

    # Team form: average of both cars per race, then rolled over prior races
    team_race = (df.groupby(["team", "season", "race_idx"])
                 .agg(t_pos=("position", "mean"), t_q=("quali_pos", "mean"), t_pts=("points", "sum"))
                 .reset_index().sort_values("race_idx"))
    tg = team_race.groupby("team")
    team_race["team_form5"] = tg.t_pos.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["team_quali5"] = tg.t_q.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["team_pts_rate"] = team_race.groupby(["team", "season"]).t_pts.transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.merge(team_race[["team", "race_idx", "team_form5", "team_quali5", "team_pts_rate"]],
                  on=["team", "race_idx"], how="left")

    # Driver's average finish at this circuit in earlier seasons
    df = df.sort_values("race_idx")
    df["drv_circuit_avg"] = df.groupby(["driver", "circuit"]).position.transform(
        lambda s: s.shift(1).expanding().mean())

    df["new_regs"] = (df.season >= 2026).astype(int)
    df["season_progress"] = df["round"] / df.groupby("season")["round"].transform("max").clip(lower=22)
    return df.sort_values(["race_idx", "driver"]).reset_index(drop=True)


class WeightedRidge:
    """Impute + scale + ridge; beat gradient boosting in the walk-forward backtest."""

    def fit(self, X, y, sample_weight=None):
        self.pipe = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10))
        self.pipe.fit(X, y, ridge__sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.pipe.predict(X)


def make_model():
    return WeightedRidge()


def sample_weights(train):
    # Recent races (especially 2026 regulations) matter most
    return np.where(train.season >= 2026, 3.0, 1.0) * np.where(train.season == 2025, 1.5, 1.0)


def simulate(pred, residuals, rng, n=N_SIMS):
    """Perturb predicted positions with resampled historical errors, rank each simulation."""
    noise = rng.choice(residuals, size=(n, len(pred)))
    finishing = (pred[None, :] + noise).argsort(axis=1).argsort(axis=1) + 1
    return finishing


def backtest(df, features, target="position"):
    races = sorted(df[(df.season >= 2025) & df[target].notna()].race_idx.unique())
    rows, residuals = [], {}
    for ridx in races:
        train = df[(df.race_idx < ridx) & df[target].notna() & df.drv_form5.notna()]
        test = df[(df.race_idx == ridx) & df[target].notna()].copy()
        m = make_model().fit(train[features], train[target], sample_weight=sample_weights(train))
        test["pred"] = m.predict(test[features])
        residuals[ridx] = (test[target] - test.pred).to_numpy()
        # Baseline: race = season points-per-race (recent form as tiebreak); quali = recent quali average
        if target == "quali_pos":
            test["base"] = test.drv_quali5.fillna(20)
        else:
            test["base"] = -test.drv_pts_rate.fillna(0) + test.drv_form5.fillna(20) / 100
        actual_winner = test.loc[test[target].idxmin(), "driver"]
        actual_podium = set(test.nsmallest(3, target).driver)
        rows.append({
            "race": ridx,
            "spearman_model": spearmanr(test.pred, test[target]).statistic,
            "spearman_base": spearmanr(test.base, test[target]).statistic,
            "winner_model": test.loc[test.pred.idxmin(), "driver"] == actual_winner,
            "winner_base": test.loc[test.base.idxmin(), "driver"] == actual_winner,
            "podium_model": len(set(test.nsmallest(3, "pred").driver) & actual_podium) / 3,
            "podium_base": len(set(test.nsmallest(3, "base").driver) & actual_podium) / 3,
            "mae_model": (test.pred - test[target]).abs().mean(),
        })
    return pd.DataFrame(rows), residuals


def report_backtest(bt, label):
    def block(sub, name):
        print(f"  {name:<14} races={len(sub):>2}  "
              f"rank corr {sub.spearman_model.mean():.3f} (baseline {sub.spearman_base.mean():.3f})  "
              f"winner {sub.winner_model.mean():.0%} ({sub.winner_base.mean():.0%})  "
              f"podium {sub.podium_model.mean():.0%} ({sub.podium_base.mean():.0%})  "
              f"MAE {sub.mae_model.mean():.2f}")
    base = "recent-quali-average" if label == "qualifying" else "points-per-race"
    print(f"\nWalk-forward backtest [{label}] — model vs. {base} baseline (winner = pole for qualifying)")
    block(bt[bt.race < 202600], "2025 season")
    block(bt[bt.race >= 202600], "2026 season")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, help="file with driverIds in grid order")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--quali", action="store_true", help="predict qualifying order instead of the race")
    args = ap.parse_args()

    grid = None
    if args.grid:
        grid = [l.strip() for l in args.grid.read_text().splitlines() if l.strip()]
    if args.quali and grid:
        raise SystemExit("--quali predicts qualifying, so it cannot take a --grid")
    target_col = "quali_pos" if args.quali else "position"
    features = GRID_FEATURES if grid else FORM_FEATURES
    label = "qualifying" if args.quali else "post-qualifying" if grid else "pre-qualifying"
    out_name = "quali" if args.quali else "postquali" if grid else "prequali"

    df = build_features(append_target_race(load(), grid))
    hist = df[df[target_col].notna()]

    bt, residuals = backtest(hist, features, target_col)
    if args.backtest:
        report_backtest(bt, label)

    train = hist[hist.drv_form5.notna()]
    model = make_model().fit(train[features], train[target_col], sample_weight=sample_weights(train))
    target = df[df.race_idx == TARGET_SEASON * 100 + TARGET_ROUND].copy()
    target["pred"] = model.predict(target[features])

    # Only use 2026 residuals for noise: they reflect this season's unpredictability
    res_2026 = np.concatenate([r for ridx, r in residuals.items() if ridx >= 202600])
    sims = simulate(target.pred.to_numpy(), res_2026, np.random.default_rng(7))
    target["win"] = (sims == 1).mean(axis=0)
    target["podium"] = (sims <= 3).mean(axis=0)
    target["points"] = (sims <= 10).mean(axis=0)
    target["exp_pos"] = sims.mean(axis=0)
    out = target.sort_values("exp_pos").reset_index(drop=True)
    out.index += 1

    if args.quali:
        # Same simulation, read as pole / front row / top-3 / reaching Q3
        target = target.rename(columns={"win": "pole", "podium": "top3", "points": "reach_q3"})
        target["front_row"] = (sims <= 2).mean(axis=0)
        out = target.sort_values("exp_pos").reset_index(drop=True)
        out.index += 1

    print(f"\n2026 Qatar Grand Prix — {label} prediction ({N_SIMS:,} simulations)\n")
    probs = ["pole", "front_row", "top3", "reach_q3"] if args.quali else ["win", "podium", "points"]
    cols = ["driver_name", "team_name"] + (["quali_pos"] if grid else []) + ["exp_pos"] + probs
    show = out[cols].copy()
    show["exp_pos"] = show.exp_pos.round(1)
    for c in probs:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print(show.to_string())
    out.to_csv(Path(__file__).parent / f"prediction_{out_name}.csv")


if __name__ == "__main__":
    main()
