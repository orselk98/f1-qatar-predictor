# F1 Qatar GP 2026 Predictor

Predicts the finishing order of the 2026 Qatar Grand Prix (Lusail, Round 22, 29 Nov 2026) from 2022–2026 race and qualifying data ([Jolpica F1 API](https://github.com/jolpica/jolpica-f1)).

## How it works
1. **Features** (every one computed only from races *before* the race being predicted, so no leakage):
   driver/team rolling finish and qualifying averages (last 5 races), points per race this season,
   DNF rate (last 10), the driver's past results at the circuit, a 2026-regulations flag, and qualifying position (post-quali model only).
2. **Model**: ridge regression on finishing position, with 2026 races weighted 3×. It beat gradient boosting in the backtest.
3. **Simulation**: 20,000 Monte Carlo races add resampled 2026 prediction errors to each driver's predicted position, giving win, podium and points probabilities.

## Usage
```bash
.venv/bin/python fetch_data.py            # refresh data (current season always refetched)
.venv/bin/python predict.py --backtest    # pre-qualifying prediction + walk-forward evaluation
.venv/bin/python predict.py --quali       # Saturday qualifying: pole / front row / top-3 / Q3 probabilities
.venv/bin/python predict.py --grid grid.txt   # after qualifying: driverIds in qualifying order, one per line
```
### Race extras: pole time, podium, safety car, rain, DNFs
```bash
.venv/bin/python fetch_events.py   # SC/VSC/rain history per race from OpenF1 (resumable)
.venv/bin/python race_extras.py
```
- **Pole time**: Lusail 2025 pole × median 2026/2025 pole-time ratio over the last 6 shared dry circuits (2026 cars gained pace through the season).
- **Safety car**: smoothed blend of 2026 rate (50%), Lusail history (30%), 2023–25 base rate (20%).
- **Rain**: Open-Meteo forecast for the race window, plus Lusail race history.
- **DNFs**: negative binomial fitted to 2026 DNFs per race (75%) blended with Lusail history (25%).
- **Qualifying top 3**: top 3 from `predict.py --quali` (same features and ridge model, target = qualifying position).
- **Podium**: top 3 from `predict.py` (uses the post-qualifying file if present).

Use driverIds as they appear in `data/results.csv` (e.g. `antonelli`, `max_verstappen`, `russell`, `norris`).

## Backtest (walk-forward: train on all earlier races, predict the next one)
| Model | Season | Rank corr. | Winner correct | Podium overlap |
|---|---|---|---|---|
| Pre-qualifying | 2026 | 0.64 (baseline 0.64) | 29% (33%) | 46% (52%) |
| Qualifying (pole) | 2026 | 0.83 (baseline 0.80) | 19% (24%) | 41% (40%) |
| Post-qualifying | 2026 | 0.69 (baseline 0.64) | 52% (33%) | 62% (52%) |

Baseline = rank drivers by points per race this season. Before qualifying, the model only matches the baseline. After qualifying it clearly beats it.
