#!/usr/bin/env python3
"""Rank all observed T20/T20I players using recent performance and pressure context.

Procedure
---------
1. Read delivery, match, competition, team, playing-XI, and player-mapping CSVs.
2. Keep all T20 and T20I competition matches dated from 2022 onward.
3. Reconstruct each innings' score and wickets before every delivery.
4. Mark shared pressure deliveries, batter-only recovery balls, and pre-delivery
   powerplay partnership context for identifying wickets of established batters.
5. Aggregate batting and bowling volume, rate, consistency proxies, and pressure stats.
6. Compare context-only HistGradientBoosting and Random Forest models using
   match-grouped out-of-fold predictions, then measure player performance
   above/below expected outcomes.
7. Apply a one-year half-life to pressure outcomes, shrink noisy player rates, and
   rank every observed player.
8. Save rankings, model diagnostics, and print the top ten in each role.

Ranking algorithm
-----------------
The final ranking uses weighted percentile ranks of match-grouped out-of-fold
context-model residuals, then multiplies the score by a reliability factor
`sqrt(n / (n + k))` (k=100 batter pressure balls; k=90 legal bowler pressure balls).
For top-order batters (batting at positions 1–3 in at least half of their observed
innings), eligibility also requires at least 10 innings, batting average >30,
boundary runs >50% of runs, overall strike rate >=150, and at least half of
innings with six or more balls faced at strike rate >=150. Top-order players who
do not meet all thresholds are kept in the full file but cannot enter the Top 10.
Openers are players who batted at positions 1–2 in at least half of their innings,
with at least 10 opening innings. At least half of those opening innings must have
30+ runs, strike rate >=160, and >=50% of runs from boundaries. This opener consistency
rate is worth 5% of the score, with non-openers receiving a neutral value.
Batting score blends shrunken pressure runs above expectation (30%), dismissals
avoided versus expectation (15%), pressure-instance consistency (10%), last-five-over
strike-rate-at-least-160 frequency (10%), first-innings early-collapse recovery runs
above expectation (5%), powerplay-collapse recovery runs (10%), middle-order pressure
runs (5%), middle-order strike-rate against spin (5%), opener consistency (5%), and
sample volume (5%). Early collapse means the first half of the first
innings with at least two wickets lost. Powerplay collapse means overs 1–6 with at
least two wickets already lost. Middle-order means batting positions 4–6; these balls
count as batter pressure after two wickets are down or under the shared pressure
definition. Both role-specific rates are shrunk with a 60-ball prior. Spin handling
uses all balls faced against spin while batting at positions 4–6; the context model
includes a spin indicator. The raw strike rate is reported, and above-expected runs
per 100 balls contributes to the rank. This component is scored only after at least
30 such balls; other players receive a neutral value. The middle-order
component is evaluated for players with at least 10 innings and at least half of those
innings at positions 4–6; other roles receive a neutral component score. The last-five-over
metric requires at least six balls faced in that phase and uses recency-weighted
successes with prior shrinkage. Bowling score blends shrunken pressure runs saved
versus expectation (30%), wickets above expectation (20%), wides/no-balls avoided
versus expectation (15%), dot balls above context expectation (10%), powerplay
wickets of established batters above expectation (5%), death-over wicket-taking
and economy against a set batter with wickets in hand (15% combined), and sample
volume (5%).
A set batter has scored at least 20 runs from 12+ balls in a partnership of 30+
runs before the delivery. This applies in overs 1–6, using legal-ball opportunities
and a 30-ball prior. Both scores are multiplied by a reliability factor
sqrt(n/(n+k)) to reduce small-sample outliers. These weights are transparent
decision rules applied to ML residuals; the model estimates context, while the
ranking score measures pressure performance beyond that context.

Machine-learning model
----------------------
HistGradientBoostingRegressor and RandomForestRegressor are compared separately
for batter runs, batter dismissals, bowler-conceded runs, bowler wickets,
wides/oversteps, and dot balls on legal deliveries. Three-fold GroupKFold holds out whole matches, preventing
deliveries from the same match leaking across train and validation. For each
outcome the lower-MAE model (or lower Brier score for binary outcomes) is used.
Player identity is excluded from features: each player's pressure result is
compared with expected results for that situation. Rankings use shrunken residual
rates and pressure consistency, not in-sample fitted outcomes. This is a
context-adjusted ranking aid, not a guaranteed forecast.

Requirements
------------
pandas, numpy, scikit-learn

Recency
-------
Match outcomes are weighted by an exponential decay with a one-year half-life,
so the latest season has more influence while older seasons still contribute.

Run from the repository root:
    python3 src/player_rankings.py
    python3 src/player_rankings.py --data-dir data --output-dir pressure_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# A chase is pressure from required rate only at this demanding rate; adjust here
# if the selected 15–17 runs/over threshold should change later.
CHASE_PRESSURE_REQUIRED_RATE = 16.0


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load supporting CSVs, canonicalize players, and filter all T20/T20I matches from 2022+."""
    balls = pd.read_csv(data_dir / "ball_by_ball.csv")
    matches = pd.read_csv(data_dir / "matches.csv")
    competitions = pd.read_csv(data_dir / "competitions.csv")
    teams = pd.read_csv(data_dir / "teams.csv")
    playing_xi = pd.read_csv(data_dir / "playing_eleven.csv")
    player_map = pd.read_csv(data_dir / "competition_team_player_mapping.csv")

    match_info = matches.merge(
        competitions[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id", how="left", validate="many_to_one",
    )
    match_info["match_date"] = pd.to_datetime(match_info["match_date"], errors="coerce")
    is_t20 = match_info["match_format"].isin(["T20", "T20I"])
    match_info = match_info[is_t20].copy()
    match_info = match_info[match_info["match_date"].dt.year.ge(2022)]

    keep = ["match_id", "comp_id", "comp_name", "match_format", "season", "match_date",
            "home_team_id", "away_team_id", "win_team_id",
            "first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "knock_out", "stage"]
    balls = balls.merge(match_info[keep], on="match_id", how="inner", validate="many_to_one")
    if balls.empty:
        raise ValueError("No T20/T20I ball-by-ball records from 2022 onward were found.")

    # Join team directory to the competition-player crosswalk, then use the
    # competition/team/player key to canonicalize player display names.
    team_names = teams[["comp_id", "team_id", "team_name"]].drop_duplicates()
    player_map = player_map.merge(
        team_names, on=["comp_id", "team_id"], how="left", validate="many_to_one",
        suffixes=("_mapping", "_directory"),
    )
    mapping_keys = ["comp_id", "team_id", "player_id"]
    if player_map.duplicated(mapping_keys).any():
        raise ValueError("Player crosswalk has duplicate competition/team/player keys.")

    bat_names = player_map[mapping_keys + ["player_name"]].rename(columns={
        "team_id": "batting_team_id", "player_id": "striker_id", "player_name": "striker_name_map",
    })
    bowl_names = player_map[mapping_keys + ["player_name"]].rename(columns={
        "team_id": "bowling_team_id", "player_id": "bowler_id", "player_name": "bowler_name_map",
    })
    balls = balls.merge(bat_names, on=["comp_id", "batting_team_id", "striker_id"],
                        how="left", validate="many_to_one")
    balls = balls.merge(bowl_names, on=["comp_id", "bowling_team_id", "bowler_id"],
                        how="left", validate="many_to_one")
    balls["striker_name"] = balls["striker_name_map"].fillna(balls["striker_name"])
    balls["bowler_name"] = balls["bowler_name_map"].fillna(balls["bowler_name"])
    balls = balls.drop(columns=["striker_name_map", "bowler_name_map"])

    # Attach player roles/positions from the playing-XI file. A few matches have
    # duplicate player rows, so collapse to one record per match/player first.
    xi = playing_xi[playing_xi["match_id"].isin(match_info["match_id"])].copy()
    xi["batting_order"] = pd.to_numeric(xi["batting_order"], errors="coerce")
    xi = xi.groupby(["match_id", "player_id"], as_index=False).agg(
        batting_order=("batting_order", "min"), is_captain=("is_captain", "max"),
        is_wk=("is_wk", "max"), is_played=("is_played", "max"),
    )
    batter_xi = xi.rename(columns={
        "player_id": "striker_id", "batting_order": "striker_batting_order",
        "is_captain": "striker_is_captain", "is_wk": "striker_is_wk",
        "is_played": "striker_is_played",
    })
    balls = balls.merge(batter_xi, on=["match_id", "striker_id"], how="left", validate="many_to_one")
    bowler_xi = xi.rename(columns={
        "player_id": "bowler_id", "batting_order": "bowler_batting_order",
        "is_captain": "bowler_is_captain", "is_wk": "bowler_is_wk",
        "is_played": "bowler_is_played",
    })
    balls = balls.merge(bowler_xi, on=["match_id", "bowler_id"], how="left", validate="many_to_one")

    # Resolve team names from the supplied competition-specific team directory.
    home_names = team_names.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"})
    away_names = team_names.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"})
    balls = balls.merge(home_names, on=["comp_id", "home_team_id"], how="left", validate="many_to_one")
    balls = balls.merge(away_names, on=["comp_id", "away_team_id"], how="left", validate="many_to_one")
    return balls, match_info


def add_context(balls: pd.DataFrame) -> pd.DataFrame:
    """Add pre-delivery score state and a reproducible pressure indicator."""
    b = balls.sort_values(["match_id", "innings", "innings_ball_num"]).copy()
    b["runs"] = pd.to_numeric(b["runs"], errors="coerce").fillna(0)
    b["is_wicket"] = b["is_wicket"].fillna(False).astype(bool)
    b["is_extra"] = b["extras_type"].notna()
    b["legal_delivery"] = ~b["extras_type"].isin(["wide", "Overstep"])
    # In this source `extras1_runs` is an extra-type code, not an amount;
    # `extras_2_runs` contains the amount but is zero/missing for many wides and
    # no-balls. Use the recorded amount when present and a one-run minimum when
    # only the extra type is known. Batter `runs` remain as supplied.
    extra_amount = pd.to_numeric(b["extras_2_runs"], errors="coerce").fillna(0).clip(lower=0)
    has_extra_type = b["extras_type"].notna()
    b["estimated_extra_runs"] = np.where(has_extra_type, np.maximum(extra_amount, 1), 0)
    b["total_runs_est"] = b["runs"] + b["estimated_extra_runs"]
    group = b.groupby(["match_id", "innings"], sort=False)
    b["score_before"] = group["total_runs_est"].cumsum() - b["total_runs_est"]
    b["wickets_before"] = group["is_wicket"].cumsum() - b["is_wicket"].astype(int)

    # Partnership state is pre-delivery and resets after each wicket. Wides do
    # not count as faced balls; no-balls do.
    partnership_keys = ["match_id", "innings", "wickets_before"]
    b["partnership_runs_before"] = (
        b.groupby(partnership_keys, sort=False)["total_runs_est"].cumsum() - b["total_runs_est"]
    )
    batter_partnership_keys = partnership_keys + ["striker_id"]
    b["striker_runs_in_partnership_before"] = (
        b.groupby(batter_partnership_keys, sort=False)["runs"].cumsum() - b["runs"]
    )
    faced_ball = ~b["extras_type"].eq("wide")
    b["striker_balls_in_partnership_before"] = (
        b.assign(_faced_ball=faced_ball.astype(int))
        .groupby(batter_partnership_keys, sort=False)["_faced_ball"].cumsum()
        - faced_ball.astype(int)
    )

    first_innings_runs = (b[b["innings"].eq(1)].groupby("match_id")["total_runs_est"].sum())
    first_target = first_innings_runs.add(1)
    revised = b.drop_duplicates("match_id").set_index("match_id")["revised_target"]
    target = revised.where(revised.notna(), first_target)
    b["target"] = b["match_id"].map(target)

    # Respect the revised length for the current innings. A second-innings DLS
    # adjustment must not redefine the first innings' death or recovery phase.
    first_overs = pd.to_numeric(b["first_inn_revised_overs"], errors="coerce")
    second_overs = pd.to_numeric(b["second_inn_revised_overs"], errors="coerce")
    innings_number = pd.to_numeric(b["innings"], errors="coerce")
    max_overs = first_overs.where(innings_number.eq(1), second_overs).fillna(20).clip(lower=1)
    legal_balls_before = (
        b.groupby(["match_id", "innings"], sort=False)["legal_delivery"].cumsum()
        - b["legal_delivery"].astype(int)
    )
    b["balls_left"] = (max_overs * 6 - legal_balls_before).clip(lower=1)
    # Required rate is only defined for a chase. First-innings balls must not be
    # assigned a fictional target based on the innings' eventual final score.
    chase = innings_number.eq(2)
    b["runs_required"] = (b["target"] - b["score_before"]).where(chase)
    b["required_rate"] = (6 * b["runs_required"] / b["balls_left"]).replace([np.inf, -np.inf], np.nan)
    b["last_five_overs"] = pd.to_numeric(b["overid"], errors="coerce").gt(max_overs - 5)
    over_number = pd.to_numeric(b["overid"], errors="coerce")
    early_innings_half = over_number.le(np.ceil(max_overs / 2))
    b["early_wicket_recovery"] = b["innings"].eq(1) & early_innings_half & b["wickets_before"].ge(2)
    b["death_over"] = b["last_five_overs"]
    stage_knockout = b["stage"].fillna("").astype(str).str.contains(
        r"(?i)\b(?:semi[\s-]*final|final|eliminator|qualifier)\b", regex=True
    )
    b["knockout"] = pd.to_numeric(b["knock_out"], errors="coerce").fillna(0).eq(1) | stage_knockout
    b["batting_is_home"] = b["batting_team_id"].eq(b["home_team_id"]).astype(int)
    b["bowling_is_home"] = b["bowling_team_id"].eq(b["home_team_id"]).astype(int)
    chase_pressure = b["innings"].eq(2) & (
        b["required_rate"].ge(CHASE_PRESSURE_REQUIRED_RATE) | b["wickets_before"].ge(5)
    )
    b["pressure"] = b["death_over"] | chase_pressure | b["knockout"]
    b["powerplay_collapse_recovery"] = over_number.le(6) & b["wickets_before"].ge(2)
    b["powerplay_set_partnership"] = (
        over_number.le(6) & b["partnership_runs_before"].ge(30)
        & b["striker_runs_in_partnership_before"].ge(20)
        & b["striker_balls_in_partnership_before"].ge(12)
    )
    b["death_settled_batter_pressure"] = (
        b["last_five_overs"] & b["wickets_before"].le(4)
        & b["partnership_runs_before"].ge(30)
        & b["striker_runs_in_partnership_before"].ge(20)
        & b["striker_balls_in_partnership_before"].ge(12)
    )
    batting_order = pd.to_numeric(b["striker_batting_order"], errors="coerce")
    b["middle_order_pressure"] = batting_order.between(4, 6) & (
        b["pressure"] | b["wickets_before"].ge(2)
    )
    # These extra signals are batter-specific; they do not redefine pressure for
    # bowlers. They still contribute to the overall batter-pressure summary.
    b["batter_pressure"] = (
        b["pressure"] | b["early_wicket_recovery"]
        | b["powerplay_collapse_recovery"] | b["middle_order_pressure"]
    )
    return b


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def fit_context_models_and_rank(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rank candidates by out-of-fold pressure results relative to context expectations.

    Models exclude player identity so they estimate the expected delivery outcome
    for the situation. GroupKFold by match prevents within-match leakage.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error, brier_score_loss
    except ImportError as exc:
        raise SystemExit(
            "The ML ranking requires scikit-learn. From the project root run "
            "`python3 -m pip install -r requirements.txt`; inside a notebook kernel, "
            "run `%pip install -r ../requirements.txt` from src, then restart the kernel."
        ) from exc

    frame = b.copy().sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    latest_date = frame["match_date"].max()
    age_years = (latest_date - frame["match_date"]).dt.days / 365.25
    frame["recency_weight"] = np.power(0.5, age_years.clip(lower=0) / 1.0)
    frame["match_year"] = frame["match_date"].dt.year.fillna(frame["match_date"].dt.year.median()).fillna(2022)
    frame["is_ipl"] = frame["comp_name"].fillna("").str.match(r"(?i)^IPL\b").astype(int)
    frame["is_t20i"] = frame["match_format"].eq("T20I").astype(int)
    frame["is_spin"] = frame["pace_or_spin"].eq("Spin").astype(int)
    frame["batter_faced"] = ~frame["extras_type"].eq("wide")
    frame["bowler_legal"] = ~frame["extras_type"].isin(["wide", "Overstep"])
    # Bowler-conceded runs exclude byes/leg-byes. The amount field is incomplete
    # for wides/no-balls, so apply the same documented one-run minimum there.
    frame["bowl_runs_actual"] = frame["runs"] + np.where(
        frame["extras_type"].isin(["wide", "Overstep"]),
        np.maximum(pd.to_numeric(frame["extras_2_runs"], errors="coerce").fillna(0), 1), 0,
    )
    frame["bowler_extra_actual"] = frame["extras_type"].isin(["wide", "Overstep"]).astype(int)
    # Count a dot only when the legal delivery has no batter runs and no extras;
    # the supplied files do not consistently include bye/leg-bye run amounts.
    frame["bowler_dot_actual"] = frame["bowler_legal"] & frame["runs"].eq(0) & ~frame["is_extra"]
    frame["batter_out_actual"] = frame["is_wicket"] & frame["dismissed_player_id"].eq(frame["striker_id"])
    frame["bowler_wicket_actual"] = frame["is_wicket"] & ~frame["dismissal_type"].isin(
        ["Run out", "RunOut", "Obstructing the field", "Obstracting the field",
         "Retired hurt", "Retired out", "Retire Out"]
    )
    features = ["innings", "overid", "over_ball_num", "score_before", "wickets_before",
                "balls_left", "runs_required", "required_rate", "pressure", "early_wicket_recovery",
                "powerplay_collapse_recovery", "middle_order_pressure",
                "partnership_runs_before", "striker_runs_in_partnership_before",
                "striker_balls_in_partnership_before", "powerplay_set_partnership",
                "death_settled_batter_pressure",
                "is_ipl", "is_t20i", "is_spin",
                "match_year", "striker_batting_order", "batting_is_home", "bowling_is_home"]
    X = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    groups = frame["match_id"].to_numpy()
    # Three match-grouped folds keep the model comparison practical to rerun in
    # both the standalone script and the notebook while preserving leakage control.
    n_splits = min(3, frame["match_id"].nunique())
    if n_splits < 2:
        raise ValueError("At least two matches are needed for match-grouped ML ranking.")

    def oof_predict(mask: pd.Series, target: pd.Series, label: str, binary: bool = False) -> np.ndarray:
        positions = np.flatnonzero(mask.to_numpy())
        xx = X.iloc[positions]
        yy = target.iloc[positions].astype(float).to_numpy()
        gg = groups[positions]
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(gg))))
        model_predictions: dict[str, np.ndarray] = {}
        for model_name in ["HistGradientBoosting", "RandomForest"]:
            pred = np.full(len(positions), np.nan, dtype=float)
            for train_i, test_i in splitter.split(xx, yy, gg):
                if model_name == "HistGradientBoosting":
                    model = HistGradientBoostingRegressor(
                        max_iter=160, learning_rate=0.05, max_leaf_nodes=15,
                        min_samples_leaf=40, l2_regularization=2.0, random_state=42,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=30, max_depth=12, min_samples_leaf=30,
                        max_features=0.8, n_jobs=2, random_state=42,
                    )
                model.fit(xx.iloc[train_i], yy[train_i])
                pred[test_i] = model.predict(xx.iloc[test_i])
            pred = np.clip(pred, 0, 1) if binary else np.maximum(pred, 0)
            model_predictions[model_name] = pred

        def loss(pred: np.ndarray) -> float:
            return brier_score_loss(yy, pred) if binary else mean_absolute_error(yy, pred)

        losses = {name: loss(pred) for name, pred in model_predictions.items()}
        selected = min(losses, key=losses.get)
        pred = model_predictions[selected]
        metric_name = "Brier" if binary else "MAE"
        diagnostic = (f"{label}: selected {selected}; "
                      + ", ".join(f"{name} {metric_name}={value:.4f}" for name, value in losses.items()))
        frame.loc[frame.index[positions], label + "_expected"] = pred
        return pred, diagnostic

    batter_mask = frame["batter_faced"]
    _, bat_runs_diag = oof_predict(batter_mask, frame["runs"], "batter_runs")
    _, bat_out_diag = oof_predict(batter_mask, frame["batter_out_actual"].astype(int), "batter_out", binary=True)
    _, bowl_runs_diag = oof_predict(pd.Series(True, index=frame.index), frame["bowl_runs_actual"], "bowler_runs")
    _, bowl_wkt_diag = oof_predict(pd.Series(True, index=frame.index), frame["bowler_wicket_actual"].astype(int), "bowler_wicket", binary=True)
    _, bowl_extra_diag = oof_predict(pd.Series(True, index=frame.index), frame["bowler_extra_actual"], "bowler_extra", binary=True)
    _, bowl_dot_diag = oof_predict(frame["bowler_legal"], frame["bowler_dot_actual"].astype(int), "bowler_dot", binary=True)

    def pct_high(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(0).rank(method="average", pct=True)

    def pct_low(s: pd.Series) -> pd.Series:
        v = pd.to_numeric(s, errors="coerce")
        worst = (v.max(skipna=True) if v.notna().any() else 0) + 1
        return (-v.fillna(worst)).rank(method="average", pct=True)

    # Rank by stable player IDs so separate people with the same display name
    # are not accidentally merged. Names remain the presentation labels.
    batter_candidates = frame[["striker_id", "striker_name"]].dropna(subset=["striker_id"]).drop_duplicates("striker_id")
    bowler_candidates = frame[["bowler_id", "bowler_name"]].dropna(subset=["bowler_id"]).drop_duplicates("bowler_id")

    # Top-order standards use all observed batting innings, not only pressure
    # balls. Boundary contribution is the share of a player's runs from fours
    # and sixes; strike-rate consistency is the share of 6+ ball innings at 150+.
    batting_innings = frame[frame["batter_faced"]].groupby(
        ["striker_id", "match_id", "innings"], sort=False
    ).agg(
        innings_runs=("runs", "sum"), innings_balls=("batter_faced", "sum"),
        innings_outs=("batter_out_actual", "sum"), fours=("is_four", "sum"),
        sixes=("is_six", "sum"), batting_order=("striker_batting_order", "min"),
    )
    batting_innings["innings_sr"] = 100 * safe_divide(
        batting_innings["innings_runs"], batting_innings["innings_balls"]
    )
    batting_innings["innings_boundary_runs"] = 4 * batting_innings["fours"] + 6 * batting_innings["sixes"]
    batting_innings["innings_boundary_share"] = safe_divide(
        batting_innings["innings_boundary_runs"], batting_innings["innings_runs"]
    )
    batting_innings["is_top_order_innings"] = batting_innings["batting_order"].le(3)
    batting_innings["is_middle_order_innings"] = batting_innings["batting_order"].between(4, 6)
    batting_innings["is_opener_innings"] = batting_innings["batting_order"].between(1, 2)
    batting_innings["sr_eligible"] = batting_innings["innings_balls"].ge(6)
    batting_innings["sr150_success"] = (
        batting_innings["sr_eligible"] & batting_innings["innings_sr"].ge(150)
    )
    batting_innings["opener_30_sr160_boundary_success"] = (
        batting_innings["is_opener_innings"]
        & batting_innings["innings_runs"].ge(30)
        & batting_innings["innings_sr"].ge(160)
        & batting_innings["innings_boundary_share"].ge(0.5)
    )
    top_order_stats = batting_innings.groupby("striker_id").agg(
        batting_innings=("innings_runs", "size"),
        top_order_innings=("is_top_order_innings", "sum"),
        middle_order_innings=("is_middle_order_innings", "sum"),
        opener_innings=("is_opener_innings", "sum"),
        opener_qualifying_innings=("opener_30_sr160_boundary_success", "sum"),
        career_runs=("innings_runs", "sum"), career_balls=("innings_balls", "sum"),
        career_outs=("innings_outs", "sum"), boundary_fours=("fours", "sum"),
        boundary_sixes=("sixes", "sum"),
        sr_eligible_innings=("sr_eligible", "sum"), sr150_innings=("sr150_success", "sum"),
    )
    top_order_stats["top_order_share"] = safe_divide(
        top_order_stats["top_order_innings"], top_order_stats["batting_innings"]
    )
    top_order_stats["top_order_batter"] = top_order_stats["top_order_share"].ge(0.5)
    top_order_stats["middle_order_share"] = safe_divide(
        top_order_stats["middle_order_innings"], top_order_stats["batting_innings"]
    )
    top_order_stats["middle_order_batter"] = (
        top_order_stats["middle_order_share"].ge(0.5)
        & top_order_stats["batting_innings"].ge(10)
    )
    top_order_stats["opener_share"] = safe_divide(
        top_order_stats["opener_innings"], top_order_stats["batting_innings"]
    )
    top_order_stats["opener_batter"] = (
        top_order_stats["opener_share"].ge(0.5)
    )
    top_order_stats["opener_consistency_rate"] = safe_divide(
        top_order_stats["opener_qualifying_innings"], top_order_stats["opener_innings"]
    )
    top_order_stats["opener_qualified"] = (
        top_order_stats["opener_batter"]
        & top_order_stats["opener_innings"].ge(10)
        & top_order_stats["opener_consistency_rate"].ge(0.5)
    )
    top_order_stats["batting_average"] = safe_divide(
        top_order_stats["career_runs"], top_order_stats["career_outs"]
    )
    top_order_stats["overall_strike_rate"] = 100 * safe_divide(
        top_order_stats["career_runs"], top_order_stats["career_balls"]
    )
    top_order_stats["boundary_run_share"] = safe_divide(
        4 * top_order_stats["boundary_fours"] + 6 * top_order_stats["boundary_sixes"],
        top_order_stats["career_runs"],
    )
    top_order_stats["sr150_consistency"] = safe_divide(
        top_order_stats["sr150_innings"], top_order_stats["sr_eligible_innings"]
    )
    top_order_stats["top_order_qualified"] = (
        top_order_stats["top_order_batter"]
        & top_order_stats["batting_innings"].ge(10)
        & top_order_stats["batting_average"].gt(30)
        & top_order_stats["boundary_run_share"].gt(0.5)
        & top_order_stats["overall_strike_rate"].ge(150)
        & top_order_stats["sr150_consistency"].ge(0.5)
        & (~top_order_stats["opener_batter"] | top_order_stats["opener_qualified"])
    )

    # Batting: above-expected runs and dismissals avoided during pressure.
    pb = frame[frame["batter_pressure"].astype(bool) & frame["batter_faced"]].copy()
    pb["runs_above_expected"] = pb["runs"] - pb["batter_runs_expected"]
    pb["dismissals_avoided"] = pb["batter_out_expected"] - pb["batter_out_actual"].astype(int)
    pb["weighted_balls"] = pb["batter_faced"].astype(float) * pb["recency_weight"]
    pb["weighted_rae"] = pb["runs_above_expected"] * pb["recency_weight"]
    pb["weighted_dav"] = pb["dismissals_avoided"] * pb["recency_weight"]
    bat_agg = pb.groupby("striker_id").agg(
        pressure_runs=("runs", "sum"), pressure_balls=("batter_faced", "sum"),
        weighted_pressure_balls=("weighted_balls", "sum"),
        pressure_matches=("match_id", "nunique"), rae=("weighted_rae", "sum"),
        dav=("weighted_dav", "sum"), pressure_outs=("batter_out_actual", "sum"),
    )
    early_recovery = frame[frame["early_wicket_recovery"] & frame["batter_faced"]].copy()
    early_recovery["weighted_early_balls"] = early_recovery["recency_weight"]
    early_recovery["weighted_early_rae"] = (
        early_recovery["runs"] - early_recovery["batter_runs_expected"]
    ) * early_recovery["recency_weight"]
    early_recovery_agg = early_recovery.groupby("striker_id").agg(
        early_recovery_balls=("batter_faced", "sum"),
        weighted_early_balls=("weighted_early_balls", "sum"),
        weighted_early_rae=("weighted_early_rae", "sum"),
    )
    bat_agg = bat_agg.join(early_recovery_agg, how="left")
    powerplay_recovery = frame[
        frame["powerplay_collapse_recovery"] & frame["batter_faced"]
    ].copy()
    powerplay_recovery["weighted_powerplay_balls"] = powerplay_recovery["recency_weight"]
    powerplay_recovery["weighted_powerplay_rae"] = (
        powerplay_recovery["runs"] - powerplay_recovery["batter_runs_expected"]
    ) * powerplay_recovery["recency_weight"]
    powerplay_agg = powerplay_recovery.groupby("striker_id").agg(
        powerplay_collapse_balls=("batter_faced", "sum"),
        weighted_powerplay_balls=("weighted_powerplay_balls", "sum"),
        weighted_powerplay_rae=("weighted_powerplay_rae", "sum"),
    )
    bat_agg = bat_agg.join(powerplay_agg, how="left")
    middle_order = frame[frame["middle_order_pressure"] & frame["batter_faced"]].copy()
    middle_order["weighted_middle_order_balls"] = middle_order["recency_weight"]
    middle_order["weighted_middle_order_rae"] = (
        middle_order["runs"] - middle_order["batter_runs_expected"]
    ) * middle_order["recency_weight"]
    middle_order_agg = middle_order.groupby("striker_id").agg(
        middle_order_pressure_balls=("batter_faced", "sum"),
        weighted_middle_order_balls=("weighted_middle_order_balls", "sum"),
        weighted_middle_order_rae=("weighted_middle_order_rae", "sum"),
    )
    bat_agg = bat_agg.join(middle_order_agg, how="left")
    # Measure middle-order performance against spin across all innings. The
    # context model includes is_spin, so this is above/below expected, not raw
    # strike-rate comparison across dissimilar deliveries.
    middle_order_spin = frame[
        frame["striker_batting_order"].between(4, 6)
        & frame["is_spin"].eq(1)
        & frame["batter_faced"]
    ].copy()
    middle_order_spin["weighted_spin_balls"] = middle_order_spin["recency_weight"]
    middle_order_spin["weighted_spin_rae"] = (
        middle_order_spin["runs"] - middle_order_spin["batter_runs_expected"]
    ) * middle_order_spin["recency_weight"]
    middle_order_spin_agg = middle_order_spin.groupby("striker_id").agg(
        middle_order_spin_runs=("runs", "sum"),
        middle_order_spin_balls=("batter_faced", "sum"),
        weighted_middle_order_spin_balls=("weighted_spin_balls", "sum"),
        weighted_middle_order_spin_rae=("weighted_spin_rae", "sum"),
    )
    bat_agg = bat_agg.join(middle_order_spin_agg, how="left")
    bat_inst = pb.groupby(["striker_id", "match_id"]).agg(
        net=("runs_above_expected", "sum"), recency=("recency_weight", "mean")
    )
    bat_inst["weighted_success"] = bat_inst["net"].gt(0).astype(float) * bat_inst["recency"]
    bat_cons = bat_inst.groupby(level=0).agg(success=("weighted_success", "sum"), total_weight=("recency", "sum"))
    bat_agg["consistency"] = (bat_cons["success"] + 1.5) / (bat_cons["total_weight"] + 3)
    bat_rank = batter_candidates.merge(
        bat_agg.reset_index(), on="striker_id", how="left"
    ).fillna({"pressure_runs": 0, "pressure_balls": 0, "pressure_matches": 0,
              "rae": 0, "dav": 0, "pressure_outs": 0, "consistency": 0.5,
              "early_recovery_balls": 0, "weighted_early_balls": 0, "weighted_early_rae": 0,
              "powerplay_collapse_balls": 0, "weighted_powerplay_balls": 0, "weighted_powerplay_rae": 0,
              "middle_order_pressure_balls": 0, "weighted_middle_order_balls": 0,
              "weighted_middle_order_rae": 0, "middle_order_spin_runs": 0,
              "middle_order_spin_balls": 0, "weighted_middle_order_spin_balls": 0,
              "weighted_middle_order_spin_rae": 0})
    bat_rank = bat_rank.merge(top_order_stats.reset_index(), on="striker_id", how="left")
    bat_rank["top_order_batter"] = bat_rank["top_order_batter"].fillna(False).astype(bool)
    bat_rank["top_order_qualified"] = bat_rank["top_order_qualified"].fillna(False).astype(bool)
    bat_rank["middle_order_batter"] = bat_rank["middle_order_batter"].fillna(False).astype(bool)
    bat_rank["opener_batter"] = bat_rank["opener_batter"].fillna(False).astype(bool)
    bat_rank["opener_qualified"] = bat_rank["opener_qualified"].fillna(False).astype(bool)
    for metric in ["top_order_share", "middle_order_share", "opener_share", "opener_consistency_rate", "batting_average", "overall_strike_rate", "boundary_run_share", "sr150_consistency"]:
        bat_rank[metric] = bat_rank[metric].fillna(0)
    bat_rank["pressure_average"] = safe_divide(bat_rank["pressure_runs"], bat_rank["pressure_outs"])
    bat_rank["pressure_strike_rate"] = 100 * safe_divide(bat_rank["pressure_runs"], bat_rank["pressure_balls"])
    bat_rank["weighted_pressure_balls"] = bat_rank["weighted_pressure_balls"].fillna(0)
    bat_rank["early_recovery_balls"] = bat_rank["early_recovery_balls"].fillna(0)
    bat_rank["early_recovery_rae_per100_shrunk"] = (
        100 * bat_rank["weighted_early_rae"] / (bat_rank["weighted_early_balls"] + 60)
    )
    bat_rank["powerplay_collapse_rae_per100_shrunk"] = (
        100 * bat_rank["weighted_powerplay_rae"] / (bat_rank["weighted_powerplay_balls"] + 60)
    )
    bat_rank["middle_order_pressure_rae_per100_shrunk"] = (
        100 * bat_rank["weighted_middle_order_rae"] / (bat_rank["weighted_middle_order_balls"] + 60)
    )
    bat_rank["middle_order_spin_strike_rate"] = 100 * safe_divide(
        bat_rank["middle_order_spin_runs"], bat_rank["middle_order_spin_balls"]
    )
    bat_rank["middle_order_spin_rae_per100_shrunk"] = (
        100 * bat_rank["weighted_middle_order_spin_rae"]
        / (bat_rank["weighted_middle_order_spin_balls"] + 60)
    )
    # Measure a batter's strike rate within each innings' final five overs.
    # Require at least six balls faced in that phase to reduce one-ball noise.
    last_five = frame[frame["last_five_overs"] & frame["batter_faced"]].copy()
    last_five_innings = last_five.groupby(["striker_id", "match_id", "innings"]).agg(
        runs=("runs", "sum"), balls=("batter_faced", "sum"), recency=("recency_weight", "mean")
    )
    eligible_last_five = last_five_innings[last_five_innings["balls"].ge(6)].copy()
    eligible_last_five["hit_160"] = (100 * eligible_last_five["runs"] / eligible_last_five["balls"]).ge(160).astype(float)
    eligible_last_five["weighted_hit_160"] = eligible_last_five["hit_160"] * eligible_last_five["recency"]
    last_five_summary = eligible_last_five.groupby("striker_id").agg(
        last5_sr_160_eligible_innings=("hit_160", "size"),
        last5_sr_160_weighted_successes=("weighted_hit_160", "sum"),
        last5_sr_160_weight=("recency", "sum"),
    )
    bat_rank = bat_rank.merge(last_five_summary, on="striker_id", how="left")
    bat_rank["last5_sr_160_eligible_innings"] = bat_rank["last5_sr_160_eligible_innings"].fillna(0)
    # Prior of five innings at a 50% success rate stabilizes sparse samples.
    bat_rank["last5_sr_160_rate"] = (
        bat_rank["last5_sr_160_weighted_successes"].fillna(0) + 2.5
    ) / (bat_rank["last5_sr_160_weight"].fillna(0) + 5)
    bat_rank["rae_per100_shrunk"] = 100 * bat_rank["rae"] / (bat_rank["weighted_pressure_balls"] + 100)
    bat_rank["dav_per100_shrunk"] = 100 * bat_rank["dav"] / (bat_rank["weighted_pressure_balls"] + 100)
    middle_order_component = pct_high(bat_rank["middle_order_pressure_rae_per100_shrunk"]).where(
        bat_rank["middle_order_batter"], 0.5
    )
    middle_order_spin_component = pct_high(bat_rank["middle_order_spin_rae_per100_shrunk"]).where(
        bat_rank["middle_order_batter"] & bat_rank["middle_order_spin_balls"].ge(30), 0.5
    )
    opener_component = pct_high(bat_rank["opener_consistency_rate"]).where(
        bat_rank["opener_batter"] & bat_rank["opener_innings"].ge(10), 0.5
    )
    bat_rank["score"] = (
        .30 * pct_high(bat_rank["rae_per100_shrunk"])
        + .15 * pct_high(bat_rank["dav_per100_shrunk"])
        + .10 * pct_high(bat_rank["consistency"])
        + .10 * pct_high(bat_rank["last5_sr_160_rate"])
        + .05 * pct_high(bat_rank["early_recovery_rae_per100_shrunk"])
        + .10 * pct_high(bat_rank["powerplay_collapse_rae_per100_shrunk"])
        + .05 * middle_order_component
        + .05 * middle_order_spin_component
        + .05 * opener_component
        + .05 * pct_high(bat_rank["weighted_pressure_balls"])
    )
    bat_rank["reliability"] = np.sqrt(bat_rank["weighted_pressure_balls"] / (bat_rank["weighted_pressure_balls"] + 100))
    bat_rank["score"] *= bat_rank["reliability"]
    bat_rank.loc[bat_rank["pressure_balls"].eq(0), "score"] = -1
    # Preserve middle-order players in the pool while requiring declared top-order
    # batters to meet the user's average, boundary, and consistent-SR standards.
    top_order_fail = bat_rank["top_order_batter"] & ~bat_rank["top_order_qualified"]
    bat_rank.loc[top_order_fail, "score"] = -1
    bat_rank = bat_rank.sort_values(
        ["score", "pressure_runs", "striker_name", "striker_id"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    bat_rank.insert(0, "rank", np.arange(1, len(bat_rank) + 1))

    # Bowling: runs saved, wickets above expected, and extras avoided per pressure spell.
    pl = frame[frame["pressure"].astype(bool)].copy()
    pl["runs_saved"] = pl["bowler_runs_expected"] - pl["bowl_runs_actual"]
    pl["wickets_above_expected"] = pl["bowler_wicket_actual"].astype(int) - pl["bowler_wicket_expected"]
    pl["extras_avoided"] = pl["bowler_extra_expected"] - pl["bowler_extra_actual"]
    pl["dot_balls_above_expected"] = np.where(
        pl["bowler_legal"], pl["bowler_dot_actual"].astype(float) - pl["bowler_dot_expected"].fillna(0), 0.0
    )
    pl["weighted_legal"] = pl["bowler_legal"].astype(float) * pl["recency_weight"]
    pl["weighted_saved_runs"] = pl["runs_saved"] * pl["recency_weight"]
    pl["weighted_wickets_above"] = pl["wickets_above_expected"] * pl["recency_weight"]
    pl["weighted_extras_avoided"] = pl["extras_avoided"] * pl["recency_weight"]
    pl["weighted_dot_balls_above_expected"] = pl["dot_balls_above_expected"] * pl["recency_weight"]
    bowl_agg = pl.groupby("bowler_id").agg(
        pressure_balls=("bowler_legal", "sum"), weighted_pressure_balls=("weighted_legal", "sum"), pressure_matches=("match_id", "nunique"),
        pressure_runs_conceded=("bowl_runs_actual", "sum"),
        pressure_wickets=("bowler_wicket_actual", "sum"),
        pressure_extras=("bowler_extra_actual", "sum"),
        pressure_dot_balls=("bowler_dot_actual", "sum"),
        saved_runs=("weighted_saved_runs", "sum"), wickets_above=("weighted_wickets_above", "sum"),
        extras_avoided=("weighted_extras_avoided", "sum"),
        dot_balls_above_expected=("weighted_dot_balls_above_expected", "sum"),
    )
    # Denominator is legal pressure deliveries; shrink toward the context baseline.
    bowl_rank = bowler_candidates.merge(
        bowl_agg.reset_index(), on="bowler_id", how="left"
    ).fillna(0)
    bowl_rank["pressure_economy"] = 6 * safe_divide(bowl_rank["pressure_runs_conceded"], bowl_rank["pressure_balls"])
    bowl_rank["pressure_wicket_rate"] = 6 * safe_divide(bowl_rank["pressure_wickets"], bowl_rank["pressure_balls"])
    bowl_rank["pressure_extras_rate"] = 6 * safe_divide(bowl_rank["pressure_extras"], bowl_rank["pressure_balls"])
    bowl_rank["pressure_dot_percentage"] = 100 * safe_divide(bowl_rank["pressure_dot_balls"], bowl_rank["pressure_balls"])
    bowl_rank["weighted_pressure_balls"] = bowl_rank["weighted_pressure_balls"].fillna(0)
    den = bowl_rank["weighted_pressure_balls"] + 90
    bowl_rank["saved_runs_per100_shrunk"] = 100 * bowl_rank["saved_runs"] / den
    bowl_rank["wickets_above_per100_shrunk"] = 100 * bowl_rank["wickets_above"] / den
    bowl_rank["extras_avoided_per100_shrunk"] = 100 * bowl_rank["extras_avoided"] / den
    bowl_rank["dot_balls_above_expected_per100_shrunk"] = 100 * bowl_rank["dot_balls_above_expected"] / den
    # Powerplay wicket of an established batter in a productive stand, adjusted
    # for expected wicket probability in the same pre-delivery context.
    pp_set = frame[frame["powerplay_set_partnership"] & frame["bowler_legal"]].copy()
    pp_set["pp_set_break"] = pp_set["bowler_wicket_actual"].astype(int)
    pp_set["weighted_pp_set_ball"] = pp_set["recency_weight"]
    pp_set["weighted_pp_set_break"] = pp_set["pp_set_break"] * pp_set["recency_weight"]
    pp_set["weighted_pp_set_residual"] = (
        (pp_set["pp_set_break"] - pp_set["bowler_wicket_expected"]) * pp_set["recency_weight"]
    )
    pp_set_agg = pp_set.groupby("bowler_id").agg(
        powerplay_set_partnership_opportunities=("bowler_legal", "sum"),
        weighted_pp_set_balls=("weighted_pp_set_ball", "sum"),
        powerplay_set_partnership_breaks=("pp_set_break", "sum"),
        weighted_pp_set_breaks=("weighted_pp_set_break", "sum"),
        weighted_pp_set_residual=("weighted_pp_set_residual", "sum"),
    )
    bowl_rank = bowl_rank.merge(pp_set_agg.reset_index(), on="bowler_id", how="left").fillna({
        "powerplay_set_partnership_opportunities": 0, "weighted_pp_set_balls": 0,
        "powerplay_set_partnership_breaks": 0, "weighted_pp_set_breaks": 0,
        "weighted_pp_set_residual": 0,
    })
    bowl_rank["powerplay_set_partnership_breaks_above_expected_per100_shrunk"] = (
        100 * bowl_rank["weighted_pp_set_residual"] / (bowl_rank["weighted_pp_set_balls"] + 30)
    )
    pp_break_component = pct_high(
        bowl_rank["powerplay_set_partnership_breaks_above_expected_per100_shrunk"]
    ).where(bowl_rank["powerplay_set_partnership_opportunities"].ge(6), 0.5)
    # Last-five-over performance when the batting side still has six wickets
    # and the striker is established in a productive partnership.
    death_set = frame[frame["death_settled_batter_pressure"] & frame["bowler_legal"]].copy()
    death_set["death_set_runs_saved"] = death_set["bowler_runs_expected"] - death_set["bowl_runs_actual"]
    death_set["death_set_wicket_residual"] = (
        death_set["bowler_wicket_actual"].astype(int) - death_set["bowler_wicket_expected"]
    )
    death_set["weighted_death_ball"] = death_set["recency_weight"]
    death_set["weighted_death_saved"] = death_set["death_set_runs_saved"] * death_set["recency_weight"]
    death_set["weighted_death_wicket_residual"] = (
        death_set["death_set_wicket_residual"] * death_set["recency_weight"]
    )
    death_set_agg = death_set.groupby("bowler_id").agg(
        death_settled_batter_opportunities=("bowler_legal", "sum"),
        weighted_death_set_balls=("weighted_death_ball", "sum"),
        death_settled_batter_wickets=("bowler_wicket_actual", "sum"),
        death_settled_batter_runs_conceded=("bowl_runs_actual", "sum"),
        weighted_death_set_saved=("weighted_death_saved", "sum"),
        weighted_death_set_wicket_residual=("weighted_death_wicket_residual", "sum"),
    )
    bowl_rank = bowl_rank.merge(death_set_agg.reset_index(), on="bowler_id", how="left").fillna({
        "death_settled_batter_opportunities": 0, "weighted_death_set_balls": 0,
        "death_settled_batter_wickets": 0, "death_settled_batter_runs_conceded": 0,
        "weighted_death_set_saved": 0, "weighted_death_set_wicket_residual": 0,
    })
    bowl_rank["death_settled_batter_economy"] = 6 * safe_divide(
        bowl_rank["death_settled_batter_runs_conceded"], bowl_rank["death_settled_batter_opportunities"]
    )
    bowl_rank["death_settled_batter_runs_saved_per100_shrunk"] = (
        100 * bowl_rank["weighted_death_set_saved"] / (bowl_rank["weighted_death_set_balls"] + 30)
    )
    bowl_rank["death_settled_batter_wickets_above_expected_per100_shrunk"] = (
        100 * bowl_rank["weighted_death_set_wicket_residual"] / (bowl_rank["weighted_death_set_balls"] + 30)
    )
    death_set_sample = bowl_rank["death_settled_batter_opportunities"].ge(6)
    death_wicket_component = pct_high(
        bowl_rank["death_settled_batter_wickets_above_expected_per100_shrunk"]
    ).where(death_set_sample, 0.5)
    death_economy_component = pct_high(
        bowl_rank["death_settled_batter_runs_saved_per100_shrunk"]
    ).where(death_set_sample, 0.5)
    bowl_inst = pl.groupby(["bowler_id", "match_id"]).agg(
        net=("runs_saved", "sum"), recency=("recency_weight", "mean")
    )
    bowl_inst["weighted_success"] = bowl_inst["net"].gt(0).astype(float) * bowl_inst["recency"]
    bowl_cons = bowl_inst.groupby(level=0).agg(success=("weighted_success", "sum"), total_weight=("recency", "sum"))
    consistency_map = ((bowl_cons["success"] + 1.5) / (bowl_cons["total_weight"] + 3)).to_dict()
    bowl_rank["consistency"] = bowl_rank["bowler_id"].map(consistency_map).fillna(0.5)
    bowl_rank["score"] = (
        .30 * pct_high(bowl_rank["saved_runs_per100_shrunk"])
        + .20 * pct_high(bowl_rank["wickets_above_per100_shrunk"])
        + .15 * pct_high(bowl_rank["extras_avoided_per100_shrunk"])
        + .10 * pct_high(bowl_rank["dot_balls_above_expected_per100_shrunk"])
        + .05 * pp_break_component
        + .05 * death_wicket_component
        + .10 * death_economy_component
        + .05 * pct_high(bowl_rank["weighted_pressure_balls"])
    )
    bowl_rank["reliability"] = np.sqrt(bowl_rank["weighted_pressure_balls"] / (bowl_rank["weighted_pressure_balls"] + 90))
    bowl_rank["score"] *= bowl_rank["reliability"]
    bowl_rank.loc[bowl_rank["pressure_balls"].eq(0), "score"] = -1
    bowl_rank = bowl_rank.sort_values(
        ["score", "pressure_wickets", "bowler_name", "bowler_id"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    bowl_rank.insert(0, "rank", np.arange(1, len(bowl_rank) + 1))

    return bat_rank, bowl_rank, [bat_runs_diag, bat_out_diag, bowl_runs_diag, bowl_wkt_diag, bowl_extra_diag, bowl_dot_diag]


def print_top_ten(title: str, frame: pd.DataFrame, name_col: str) -> None:
    print(f"\n{title}")
    for row in frame.head(10).itertuples(index=False):
        print(f"{row.rank:>2}. {getattr(row, name_col)} (score={row.score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder containing input CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("pressure_outputs"), help="Where ranking CSVs are written")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, match_info = read_inputs(args.data_dir)
    contextual = add_context(raw)
    bat_ranked, bowl_ranked, ml_diagnostics = fit_context_models_and_rank(contextual)

    bat_ranked.to_csv(args.output_dir / "batting_rankings.csv", index=False)
    bowl_ranked.to_csv(args.output_dir / "bowling_rankings.csv", index=False)
    pd.DataFrame({
        "Rank": np.arange(1, 11),
        "Batter": bat_ranked.head(10)["striker_name"].to_list(),
        "Batter ID": bat_ranked.head(10)["striker_id"].to_list(),
        "Bowler": bowl_ranked.head(10)["bowler_name"].to_list(),
        "Bowler ID": bowl_ranked.head(10)["bowler_id"].to_list(),
    }).to_csv(args.output_dir / "final_top10.csv", index=False)
    competitions = match_info["comp_name"].fillna("")
    is_ipl_comp = competitions.str.match(r"(?i)^IPL\b")
    ipl_count = int(is_ipl_comp.sum())
    other_t20_count = int((match_info["match_format"].eq("T20") & ~is_ipl_comp).sum())
    t20i_count = int(match_info["match_format"].eq("T20I").sum())
    print(f"Included {len(match_info)} T20/T20I matches from 2022 onward: {ipl_count} IPL, {other_t20_count} other T20, {t20i_count} T20I.")
    world_cup_matches = match_info[match_info["comp_name"].str.contains("world cup", case=False, na=False)]
    if world_cup_matches.empty:
        print("No T20 World Cup competition records were found in the supplied data.")
    diagnostics_path = args.output_dir / "context_model_diagnostics.txt"
    diagnostic_text = "\n".join(ml_diagnostics) + "\n"
    diagnostics_path.write_text(diagnostic_text, encoding="utf-8")
    (args.output_dir / "ml_diagnostics.txt").write_text(diagnostic_text, encoding="utf-8")
    print("Context-model diagnostics: " + " | ".join(ml_diagnostics))
    integration_audit = pd.DataFrame([
        {"check": "scoped_matches", "value": len(match_info)},
        {"check": "ball_by_ball_rows", "value": len(raw)},
        {"check": "unique_batters", "value": raw["striker_id"].nunique()},
        {"check": "unique_bowlers", "value": raw["bowler_id"].nunique()},
        {"check": "batter_playing_xi_order_coverage_pct", "value": 100 * raw["striker_batting_order"].notna().mean()},
        {"check": "bowler_playing_xi_coverage_pct", "value": 100 * raw["bowler_batting_order"].notna().mean()},
        {"check": "home_team_directory_coverage_pct", "value": 100 * raw["home_team_name"].notna().mean()},
        {"check": "away_team_directory_coverage_pct", "value": 100 * raw["away_team_name"].notna().mean()},
    ])
    integration_audit.to_csv(args.output_dir / "data_integration_audit.csv", index=False)
    print(f"Integrated supporting data: XI coverage {integration_audit.loc[integration_audit.check.eq('batter_playing_xi_order_coverage_pct'),'value'].iloc[0]:.1f}% batter order, "
          f"{integration_audit.loc[integration_audit.check.eq('bowler_playing_xi_coverage_pct'),'value'].iloc[0]:.1f}% bowler; "
          f"team directory {integration_audit.loc[integration_audit.check.eq('home_team_directory_coverage_pct'),'value'].iloc[0]:.1f}%.")
    print_top_ten("Top 10 batters", bat_ranked, "striker_name")
    print_top_ten("Top 10 bowlers", bowl_ranked, "bowler_name")

    print(f"\nRanked the full observed player pool: {len(bat_ranked)} batters and {len(bowl_ranked)} bowlers.")
    print(f"\nCSV outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
