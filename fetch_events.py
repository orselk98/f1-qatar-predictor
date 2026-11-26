"""Download safety car and rainfall history per race from the OpenF1 API (2023+).

Usage: python fetch_events.py
Writes data/race_events.csv with one row per race: sc / vsc / red flag counts and rain flag.
"""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
SEASONS = range(2023, 2027)
DATA = Path(__file__).parent / "data"


def get(endpoint, **params):
    for attempt in range(8):
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        time.sleep(1.5)
        if r.status_code == 404:  # OpenF1 returns 404 for "no matching rows"
            return []
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate limited on {endpoint}")


def main():
    cache = DATA / "race_events.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["session_key"])
    rows = done.to_dict("records")
    now = pd.Timestamp.now(tz="UTC")
    for season in SEASONS:
        for s in get("sessions", year=season, session_name="Race"):
            if (s.get("is_cancelled") or pd.Timestamp(s["date_end"]) > now
                    or s["session_key"] in set(done.session_key)):
                continue
            print(f"{season} {s['circuit_short_name']}")
            msgs = get("race_control", session_key=s["session_key"])
            texts = [str(m.get("message", "")).upper() for m in msgs]
            weather = get("weather", session_key=s["session_key"])
            rows.append({
                "session_key": s["session_key"],
                "season": season,
                "circuit": s["circuit_short_name"],
                "date": s["date_start"][:10],
                "sc": sum("SAFETY CAR DEPLOYED" in t and "VIRTUAL" not in t for t in texts),
                # 2026 race control abbreviates to "VSC DEPLOYED"
                "vsc": sum("VIRTUAL SAFETY CAR DEPLOYED" in t or "VSC DEPLOYED" in t for t in texts),
                "red_flag": sum("RED FLAG" in t for t in texts if "CHEQUERED" not in t),
                "rain": int(any(w.get("rainfall") for w in weather)),
            })
            pd.DataFrame(rows).to_csv(cache, index=False)  # checkpoint: resumable after rate limits
    out = pd.DataFrame(rows).sort_values("date")
    out.to_csv(cache, index=False)
    print(f"{len(out)} races saved to {cache}")


if __name__ == "__main__":
    main()
