# T20 pressure performance rankings

## Data and model

The rankings cover 883 supplied T20/T20I matches and 200,999 deliveries from 2022 onward: 371 IPL, 30 other T20, and 482 T20I. Player identity uses the supplied player ID. No T20 World Cup records are included.

The pandas and scikit-learn pipeline is in [src/player_rankings.py](src/player_rankings.py); Section 13 of [src/pressure_analytics.ipynb](src/pressure_analytics.ipynb) calls the same ranking implementation. HistGradientBoosting and Random Forest are compared using three-fold GroupKFold validation by match. HistGradientBoosting is selected for the current six outcomes. The feature set now includes whether the delivery was bowled by spin, allowing context-adjusted comparisons against different bowling styles.

## Batter pressure and spin measures

All match state is calculated before the delivery. Legal balls remaining uses revised innings length and counts legal balls because `over_ball_num` advances on wides/no-balls. Shared pressure includes the final five overs, a chase requiring 16+ runs per over, 5+ wickets lost in a chase, or a knockout. Semi-finals and finals are explicitly included using the match stage label; the supplied knockout flag also includes qualifiers and eliminators.

Batter-specific signals include first-innings recovery in the first half after 2+ wickets fall; powerplay-collapse recovery in overs 1–6 after 2+ wickets fall; and middle-order pressure for batting positions 4–6 under shared pressure or after 2+ wickets fall. These run rates are compared with model expectations and shrunk with a 60-ball prior.

Middle-order spin strike rate is the batter's runs per 100 balls faced against deliveries labeled `Spin`, while batting positions 4–6, across all innings. The rank component uses context-adjusted runs above expectation per 100 spin balls, a 60-ball prior, and requires at least 30 spin balls. It contributes 5% of the batter score. Non-middle-order players receive a neutral value. The dashboard/report also show the raw spin strike rate and spin-ball sample.

Opener standards: an opener bats at positions 1–2 in at least half of their innings, with at least 10 opening innings. At least half of those innings must include 30+ runs at SR ≥160 and at least half of runs from fours/sixes. This is a 5% component and an eligibility condition for openers. Top-order standards also require at least 10 innings, average >30, boundary run share >50%, overall SR ≥150, and 150+ SR in at least half of 6+ ball innings.

Batting weights are pressure runs above expectation 30%, dismissals avoided 15%, pressure-match consistency 10%, last-five-over innings at SR ≥160 10%, first-innings recovery 5%, powerplay recovery 10%, middle-order pressure 5%, middle-order spin handling 5%, opener consistency 5%, and sample volume 5%. Bowling weights are runs saved 30%, wickets above expectation 20%, wides/no-balls avoided 15%, dots above expectation 10%, powerplay set-partnership wickets above expected 5%, death-over wickets above expected against a set batter 5%, death-over runs saved against a set batter 10%, and sample volume 5%. Scores are relative indices, not probabilities.

A new bowler-specific powerplay signal counts legal-ball opportunities in overs 1–6 when the stand is already 30+ runs and the striker has scored 20+ runs from at least 12 balls in that partnership. A bowler-credited dismissal is a partnership break; run outs and other non-bowler dismissals are excluded. Breaks are compared with the out-of-fold expected wicket probability and reported as residual breaks per 100 qualifying balls, shrunk with a 30-ball prior. The component is neutral below six qualifying balls. Across the supplied pool, the signal covers 4,176 qualifying legal-ball opportunities and 216 bowler-credited breaks.

Death-over settled-batter pressure isolates the last five overs when the batting side has at least six wickets in hand, the partnership has reached 30+ runs, and the striker has scored 20+ runs from at least 12 balls in that stand. The rank rewards wickets above expected (5%) and runs saved versus expected (10%) on these legal deliveries, both per 100 balls with a 30-ball prior. Raw death-over economy and wicket counts are also reported; at least six qualifying balls are required for non-neutral component scores.

## Top 10 batters

| Rank | Batter | Score | Role | Opener qualifying innings | Powerplay-collapse balls | Powerplay +/100 | Middle-order pressure balls | Middle-order +/100 | Middle-order spin balls | Spin SR | Spin +/100 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Tim David (ID 6396) | 0.851 | Middle order | 0/0 | 61 | 20.7 | 1011 | 28.5 | 298 | 155.0 | 28.8 |
| 2 | Shreyas Iyer (ID 3647) | 0.842 | Middle order | 1/1 | 240 | 5.6 | 1175 | 6.8 | 492 | 153.0 | 5.9 |
| 3 | Suryakumar Yadav (ID 1699) | 0.841 | Middle order | 1/4 | 384 | 6.1 | 1778 | 13.2 | 813 | 151.3 | 10.1 |
| 4 | Heinrich Klaasen (ID 4260) | 0.768 | Middle order | 0/0 | 103 | -1.3 | 1320 | 4.1 | 574 | 173.3 | 14.8 |
| 5 | Naman Dhir (ID 37052) | 0.759 | Other role | 0/0 | 58 | 7.8 | 173 | 14.0 | 34 | 176.5 | 14.2 |
| 6 | Sherfane Rutherford (ID 6725) | 0.731 | Middle order | 0/0 | 65 | 14.3 | 802 | 1.4 | 354 | 131.6 | 4.2 |
| 7 | Glenn Phillips (ID 4853) | 0.721 | Middle order | 0/0 | 186 | -2.8 | 1093 | 9.6 | 449 | 145.2 | 13.0 |
| 8 | Daryl Mitchell (ID 4106) | 0.714 | Middle order | 0/1 | 118 | 3.4 | 848 | 6.7 | 351 | 131.9 | 4.4 |
| 9 | Cameron Green (ID 5939) | 0.711 | Middle order | 2/7 | 133 | 12.6 | 564 | 1.7 | 244 | 150.8 | 2.8 |
| 10 | Marcus Stoinis (ID 2639) | 0.698 | Middle order | 0/1 | 116 | 1.9 | 948 | 6.4 | 423 | 135.0 | -7.4 |

## Top 10 bowlers

| Rank | Bowler | Score | Pressure balls | Pressure wickets | Pressure economy | Dot-ball % | PP set breaks / balls | Death set wickets / balls | Death set economy | Death wickets above expected / 100 | Death runs saved / 100 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jason Holder (ID 2633) | 0.743 | 757 | 59 | 9.75 | 28.4% | 0 / 52 | 5 / 93 | 11.16 | -0.32 | 10.36 |
| 2 | Jofra Archer (ID 3298) | 0.732 | 422 | 31 | 9.61 | 30.6% | 3 / 19 | 3 / 48 | 10.62 | -0.16 | 17.41 |
| 3 | Sunil Narine (ID 781) | 0.707 | 361 | 22 | 6.71 | 36.8% | 2 / 39 | 2 / 15 | 6.80 | 3.27 | 15.91 |
| 4 | Josh Hazlewood (ID 830) | 0.668 | 524 | 41 | 9.15 | 35.5% | 4 / 38 | 0 / 33 | 11.64 | -1.86 | 8.19 |
| 5 | Eshan Malinga (ID 39657) | 0.661 | 393 | 28 | 8.87 | 33.6% | 3 / 35 | 3 / 34 | 10.24 | 1.12 | 12.50 |
| 6 | Dushmantha Chameera (ID 2399) | 0.653 | 403 | 27 | 9.32 | 34.0% | 2 / 42 | 6 / 35 | 12.00 | 3.38 | 3.68 |
| 7 | Varun Chakravarthy (ID 6256) | 0.632 | 573 | 43 | 8.17 | 40.1% | 1 / 60 | 2 / 31 | 14.13 | 1.51 | 2.54 |
| 8 | Rashid Khan (ID 4722) | 0.623 | 884 | 77 | 8.08 | 34.7% | 3 / 62 | 5 / 47 | 12.64 | -1.04 | -13.33 |
| 9 | Adam Zampa (ID 2636) | 0.608 | 277 | 30 | 8.53 | 34.3% | 0 / 5 | 3 / 26 | 7.85 | 0.64 | 1.22 |
| 10 | Jasprit Bumrah (ID 2933) | 0.602 | 665 | 46 | 7.43 | 35.5% | 0 / 63 | 2 / 21 | 8.00 | -0.06 | 13.74 |

## Diagnostics and source limits

- batter_runs: selected HistGradientBoosting; HistGradientBoosting MAE=1.2964, RandomForest MAE=1.3031
- batter_out: selected HistGradientBoosting; HistGradientBoosting Brier=0.0513, RandomForest Brier=0.0515
- bowler_runs: selected HistGradientBoosting; HistGradientBoosting MAE=1.2588, RandomForest MAE=1.2655
- bowler_wicket: selected HistGradientBoosting; HistGradientBoosting Brier=0.0482, RandomForest Brier=0.0484
- bowler_extra: selected HistGradientBoosting; HistGradientBoosting Brier=0.0393, RandomForest Brier=0.0394
- bowler_dot: selected HistGradientBoosting; HistGradientBoosting Brier=0.2186, RandomForest Brier=0.2194
Playing-XI coverage is 99.9% for batting order and 99.9% for bowlers; team directory coverage is 100%. Extra-run amounts are incomplete; some multi-run extras may be underestimated. No T20 World Cup records appear in the supplied data.

Reproduce with `python3 src/player_rankings.py` from the project root, or execute Section 13 in the notebook.
