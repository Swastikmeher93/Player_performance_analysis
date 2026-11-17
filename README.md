# Cricket Player Pressure Performance Analysis

A reproducible analytics project for ranking T20/T20I batters and bowlers by performance in high-pressure match situations. The project includes a pandas and scikit-learn pipeline, a Jupyter notebook, generated rankings and diagnostics, and a Next.js dashboard.

## Data coverage

The supplied CSVs cover T20 and T20I matches from 2022 onward. The pipeline joins ball-by-ball deliveries to matches, competitions, teams, player mappings, and playing XIs. The current dataset includes IPL, other T20 competitions, and T20Is. T20 World Cup matches are included only if their match and ball-by-ball records are added to `data/`.

## Run the ranking pipeline

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 src/player_rankings.py
```

The script writes full batting and bowling rankings, top-ten selections, model diagnostics, and audit files to `pressure_outputs/`.

Open `src/pressure_analytics.ipynb` to explore the data and execute the same final ranking implementation in Section 13.

## Launch the dashboard

Generate the rankings first, then run:

```bash
cd client/performance_dashboard
npm install
npm run dev
```

Open `http://localhost:3000`. To make a production build, run `npm run build` from `client/performance_dashboard`.

## Pressure method

The final ranking marks a delivery as shared pressure when it is in the last five overs, a chase requires at least 16 runs per over or has lost at least five wickets, or the match is a knockout (including semi-finals and finals).

Batter-specific pressure also includes first-innings recovery after two or more early wickets, powerplay recovery after two or more wickets, and middle-order pressure. The batter score compares runs and dismissals with situation-adjusted model expectations and adds consistency, death-over strike-rate, role-specific recovery, spin-handling, opener-consistency, and sample-size measures.

Bowler scoring compares runs conceded, wickets, extras, and dots with model expectations. It includes powerplay wickets of set batters and death-over performance against established batters with wickets in hand. Expected outcomes are produced with match-grouped out-of-fold validation; HistGradientBoosting and Random Forest are compared per outcome. Recent matches receive more weight, and small samples are shrunk toward a baseline. Scores are relative ranking indices, not probabilities or guarantees of future performance.

Detailed thresholds, weights, diagnostics, and current rankings are in `analysis_report.md` and the generated CSV files in `pressure_outputs/`.

## Repository layout

- `data/` — input match, delivery, competition, team, player, and playing-XI CSVs
- `src/player_rankings.py` — shared ranking pipeline
- `src/pressure_analytics.ipynb` — exploratory and final analytics notebook
- `pressure_outputs/` — generated rankings, diagnostics, and figures
- `client/performance_dashboard/` — Next.js dashboard
- `analysis_report.md` — current methodology and ranking report
