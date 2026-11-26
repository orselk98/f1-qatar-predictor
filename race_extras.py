"""Predict pole time, podium, safety car, rain and DNF count for the 2026 Qatar Grand Prix.

Usage: python race_extras.py   (run predict.py and predict.py --quali first; fetch_events.py for SC/rain history)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import nbinom, poisson

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).parent
LUSAIL = (25.49, 51.45)
RACE_HOURS_LOCAL = ("19", "20", "21")
RECENT_CIRCUITS = 6  # lights out 19:00 AST, ~1h30 race
FINISHED = r"^(Finished|Lapped|\+\d+ Laps?)$"
NOT_STARTED = r"^(Did not start|Withdrew|Disqualified|Did not qualify|Excluded)$"


def lap_seconds(t):
    if not isinstance(t, str) or ":" not in t:
        return np.nan
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def fmt_lap(sec):
    return f"{int(sec // 60)}:{sec % 60:06.3f}"


def pole_time(res, quali):
    """Scale Lusail's 2025 pole by the median 2026-vs-2025 pole-time ratio at shared circuits."""
    q = quali[quali.quali_pos == 1].copy()
    q["pole_s"] = q.q3.map(lap_seconds).fillna(q.q2.map(lap_seconds)).fillna(q.q1.map(lap_seconds))
    circuits = res[["season", "round", "circuit"]].drop_duplicates()
    q = q.merge(circuits, on=["season", "round"])
    wide = q.pivot_table(index="circuit", columns="season", values="pole_s", aggfunc="first")
    rounds_2026 = q[q.season == 2026].set_index("circuit")["round"]
    shared = wide[[2025, 2026]].dropna().join(rounds_2026).sort_values("round")
    ratio = shared[2026] / shared[2025]
    # Drop wet/disrupted sessions: ratios far from the median are not about car pace
    ratio = ratio[(ratio - ratio.median()).abs() < 0.04]
    # 2026 cars gained pace through the season, so only the most recent circuits reflect Qatar
    recent = ratio.tail(RECENT_CIRCUITS)
    base = wide.loc["losail", 2025]
    lo, hi = recent.min(), recent.max()
    return {"base_2025": base, "pred": base * recent.median(), "lo": base * lo, "hi": base * hi,
            "ratio": recent.median(), "ratio_all": ratio.median(), "n_circuits": len(recent),
            "circuits": ", ".join(recent.index)}


def safety_car(events):
    """Blend this season's rate, Lusail's history and the 2023-25 base rate (Laplace-smoothed)."""
    events = events.assign(sc_any=events.sc > 0, neutral=(events.sc + events.vsc) > 0)
    out = {}
    for col in ["sc_any", "neutral"]:
        season = events[events.season == 2026][col]
        track = events[events.circuit == "Lusail"][col]
        era = events[events.season < 2026][col]
        smooth = lambda s: (s.sum() + 1) / (len(s) + 2)
        out[col] = {"p": 0.5 * smooth(season) + 0.3 * smooth(track) + 0.2 * smooth(era),
                    "season": (season.sum(), len(season)), "track": (track.sum(), len(track))}
    return out


def rain(events):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
        "latitude": LUSAIL[0], "longitude": LUSAIL[1], "timezone": "Asia/Qatar",
        "hourly": "precipitation_probability,precipitation,temperature_2m,wind_speed_10m",
        "start_date": "2026-11-29", "end_date": "2026-11-29"})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    window = h[h.time.str[11:13].isin(RACE_HOURS_LOCAL)]
    track = events[events.circuit == "Lusail"]
    return {"forecast_p": window.precipitation_probability.max() / 100,
            "forecast_mm": window.precipitation.sum(),
            "temp": (window.temperature_2m.min(), window.temperature_2m.max()),
            "wind": window.wind_speed_10m.max(),
            "track_hist": (track.rain.sum(), len(track))}


def dnfs(res):
    started = res[~res.status.str.match(NOT_STARTED)]
    per_race = (started.assign(dnf=~started.status.str.match(FINISHED))
                .groupby(["season", "round", "circuit"]).dnf.sum().reset_index())
    season = per_race[per_race.season == 2026].dnf
    track = per_race[per_race.circuit == "losail"].dnf
    mean = 0.75 * season.mean() + 0.25 * track.mean()
    var = season.var()
    # DNF counts are overdispersed (crash-fests vs. clean races): use negative binomial if so
    if var > mean:
        n = mean ** 2 / (var - mean)
        dist = nbinom(n, n / (n + mean))
    else:
        dist = poisson(mean)
    pmf = {k: dist.pmf(k) for k in range(0, 13)}
    return {"mean": mean, "mode": max(pmf, key=pmf.get), "interval": (int(dist.ppf(0.1)), int(dist.ppf(0.9))),
            "pmf": pmf, "season_mean": season.mean(), "season_range": (season.min(), season.max()),
            "track": track.tolist()}


def quali_top3():
    path = ROOT / "prediction_quali.csv"
    if not path.exists():
        raise SystemExit("Run predict.py --quali first")
    return pd.read_csv(path, index_col=0).nsmallest(3, "exp_pos")


def podium():
    for name in ["prediction_postquali.csv", "prediction_prequali.csv"]:
        path = ROOT / name
        if path.exists():
            p = pd.read_csv(path, index_col=0)
            return p.nsmallest(3, "exp_pos"), name
    raise SystemExit("Run predict.py first")


def main():
    res = pd.read_csv(DATA / "results.csv")
    quali = pd.read_csv(DATA / "qualifying.csv")
    events = pd.read_csv(DATA / "race_events.csv")

    print("\n2026 QATAR GRAND PRIX — RACE PREDICTIONS\n" + "=" * 42)

    p = pole_time(res, quali)
    print(f"\nPOLE TIME (Saturday): {fmt_lap(p['pred'])}   (likely range {fmt_lap(p['lo'])} – {fmt_lap(p['hi'])})")
    print(f"  2025 pole {fmt_lap(p['base_2025'])} × 2026 pace factor {p['ratio']:.4f} "
          f"(median of last {p['n_circuits']} shared dry circuits: {p['circuits']}; season-wide {p['ratio_all']:.4f})")

    print("\nQUALIFYING TOP 3 (Saturday)")
    for i, (_, r) in enumerate(quali_top3().iterrows(), 1):
        print(f"  P{i}  {r.driver_name:<24} {r.team_name:<10} pole {r.pole:.0%}, top-3 chance {r.top3:.0%}")

    top3, src = podium()
    print(f"\nPODIUM (Sunday)   [{src}]")
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        print(f"  P{i}  {r.driver_name:<24} {r.team_name:<10} podium chance {r.podium:.0%}")

    sc = safety_car(events)
    s, n = sc["sc_any"]["season"]
    t, tn = sc["sc_any"]["track"]
    verdict = "YES" if sc["sc_any"]["p"] >= 0.5 else "NO"
    print(f"\nSAFETY CAR: {verdict}   (P = {sc['sc_any']['p']:.0%})")
    print(f"  2026: {s}/{n} races had a full SC; Lusail 2023-25: {t}/{tn}. "
          f"SC or VSC: P = {sc['neutral']['p']:.0%}")

    w = rain(events)
    rp = max(w["forecast_p"], 0.01)
    print(f"\nRAIN: {'YES' if rp >= 0.5 else 'NO'}   (P ≈ {rp:.0%})")
    print(f"  Forecast race window: {w['forecast_p']:.0%} chance, {w['forecast_mm']:.1f} mm, "
          f"{w['temp'][0]:.0f}–{w['temp'][1]:.0f}°C, wind ≤ {w['wind']:.0f} km/h. "
          f"Lusail races with rain: {w['track_hist'][0]}/{w['track_hist'][1]}")

    d = dnfs(res)
    print(f"\nDNFs: {d['mode']} most likely   (expected {d['mean']:.1f}, 80% range {d['interval'][0]}–{d['interval'][1]})")
    print(f"  2026 average {d['season_mean']:.1f}/race (range {d['season_range'][0]}–{d['season_range'][1]}); "
          f"Lusail 2023-25: {d['track']}")
    print("  " + "  ".join(f"{k}:{v:.0%}" for k, v in d["pmf"].items() if v >= 0.02))


if __name__ == "__main__":
    main()
