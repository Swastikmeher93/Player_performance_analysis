# Pressure / Play dashboard

Interactive dashboard for the generated T20 and T20I pressure rankings. The dashboard loads the current ranking CSVs from the repository's `pressure_outputs/` folder through a Next.js API route. It includes batter and bowler leaderboards, player search, and pressure-profile details.

## Run locally

From the repository root, generate current rankings if they are not already present:

```bash
python3 src/player_rankings.py
```

Then start the dashboard:

```bash
cd client/performance_dashboard
npm install
npm run dev
```

Open `http://localhost:3000`.

## Build

```bash
npm run build
npm start
```

The final ranking includes pressure in the last five overs, demanding chases, knockout stages, and role-specific batter situations. Bowling scores also include powerplay partnership breaks and wickets/economy against established batters in death overs with wickets in hand. See the repository root `analysis_report.md` for exact criteria and weights.
