"""Download race results and qualifying from the Jolpica (Ergast-compatible) F1 API.

Usage: python fetch_data.py [--refresh]
Writes data/results.csv and data/qualifying.csv.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
SEASONS = range(2022, 2027)  # ground-effect era + 2026 regulations
DATA = Path(__file__).parent / "data"
PAGE = 100


def get_paged(path):
    races, offset = {}, 0
    while True:
        for attempt in range(5):
            r = requests.get(f"{BASE}/{path}.json", params={"limit": PAGE, "offset": offset}, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        mr = r.json()["MRData"]
        # Pages split races by result row, so merge rows for the same round
        for race in mr["RaceTable"]["Races"]:
            key = (race["season"], race["round"])
            rows_key = "Results" if "Results" in race else "QualifyingResults"
            if key in races:
                races[key][rows_key].extend(race[rows_key])
            else:
                races[key] = race
        offset += PAGE
        time.sleep(0.3)
        if offset >= int(mr["total"]):
            return list(races.values())


def flatten_results(races):
    rows = []
    for race in races:
        for res in race["Results"]:
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race": race["raceName"],
                "circuit": race["Circuit"]["circuitId"],
                "date": race["date"],
                "driver": res["Driver"]["driverId"],
                "driver_name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                "team": res["Constructor"]["constructorId"],
                "team_name": res["Constructor"]["name"],
                "grid": int(res["grid"]),
                "position": int(res["position"]),
                "points": float(res["points"]),
                "status": res["status"],
            })
    return pd.DataFrame(rows)


def flatten_quali(races):
    rows = []
    for race in races:
        for res in race["QualifyingResults"]:
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "driver": res["Driver"]["driverId"],
                "quali_pos": int(res["position"]),
                "q1": res.get("Q1"),
                "q2": res.get("Q2"),
                "q3": res.get("Q3"),
            })
    return pd.DataFrame(rows)


def main():
    DATA.mkdir(exist_ok=True)
    refresh = "--refresh" in sys.argv
    res_frames, q_frames = [], []
    for season in SEASONS:
        res_cache, q_cache = DATA / f"results_{season}.csv", DATA / f"quali_{season}.csv"
        # Always refetch the current season; past seasons are immutable
        if res_cache.exists() and not refresh and season < max(SEASONS):
            res_frames.append(pd.read_csv(res_cache))
            q_frames.append(pd.read_csv(q_cache))
            continue
        print(f"Fetching {season}...")
        res = flatten_results(get_paged(f"{season}/results"))
        q = flatten_quali(get_paged(f"{season}/qualifying"))
        res.to_csv(res_cache, index=False)
        q.to_csv(q_cache, index=False)
        res_frames.append(res)
        q_frames.append(q)
    results = pd.concat(res_frames, ignore_index=True)
    quali = pd.concat(q_frames, ignore_index=True)
    results.to_csv(DATA / "results.csv", index=False)
    quali.to_csv(DATA / "qualifying.csv", index=False)
    print(f"{len(results)} result rows, {len(quali)} qualifying rows, "
          f"latest: {results.season.max()} R{results[results.season == results.season.max()]['round'].max()}")


if __name__ == "__main__":
    main()
