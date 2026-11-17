"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

type RankingRow = Record<string, number | string>;
type DashboardData = {
  batters: RankingRow[];
  bowlers: RankingRow[];
  audit: RankingRow[];
  diagnostics: string[];
  updatedAt: string;
};
type Role = "batters" | "bowlers";
const EMPTY_ROWS: RankingRow[] = [];

const PLAYER_FIELD: Record<Role, string> = {
  batters: "striker_name",
  bowlers: "bowler_name",
};

function value(row: RankingRow | undefined, key: string): number {
  const result = Number(row?.[key] ?? 0);
  return Number.isFinite(result) ? result : 0;
}

function flag(row: RankingRow | undefined, key: string): boolean {
  const result = row?.[key];
  return result === 1 || String(result).toLowerCase() === "true" || String(result) === "1";
}

function battingRoleStatus(row: RankingRow): string {
  if (flag(row, "opener_batter")) return flag(row, "opener_qualified") ? "OPENER · PASS" : "OPENER · NOT MET";
  if (flag(row, "top_order_batter")) return flag(row, "top_order_qualified") ? "TOP ORDER · PASS" : "TOP ORDER · NOT MET";
  if (flag(row, "middle_order_batter")) return "MIDDLE ORDER";
  return "OTHER ROLE";
}

function label(row: RankingRow | undefined, role: Role): string {
  return String(row?.[PLAYER_FIELD[role]] ?? "—");
}

function identity(row: RankingRow | undefined, role: Role): string {
  const idField = role === "batters" ? "striker_id" : "bowler_id";
  return String(row?.[idField] ?? label(row, role));
}

function playerLabel(row: RankingRow, role: Role, rows: RankingRow[]): string {
  const name = label(row, role);
  const duplicates = rows.filter((candidate) => label(candidate, role) === name).length > 1;
  return duplicates ? `${name} · ID ${identity(row, role)}` : name;
}

function formatNumber(number: number): string {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(number);
}

function formatPercent(number: number, fraction = false): string {
  const percentage = fraction ? number * 100 : number;
  return `${percentage.toFixed(1)}%`;
}

function auditValue(audit: RankingRow[], key: string): number {
  const result = audit.find((row) => row.check === key);
  return value(result, "value");
}

async function requestDashboardData(): Promise<DashboardData> {
  const response = await fetch("/api/rankings", { cache: "no-store" });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? "Could not load rankings.");
  return result as DashboardData;
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 36 36" fill="none">
        <path d="M18 3.5 31 11v14L18 32.5 5 25V11L18 3.5Z" stroke="currentColor" strokeWidth="1.6" />
        <path d="M11 21.5V14l7-4 7 4v7.5l-7 4-7-4Z" fill="currentColor" />
        <path d="m14.5 18 2.2 2.2 4.8-5" stroke="#F9F8F4" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

function TrendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 17.5 9 12l3.2 3.2L20 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M15.5 7.5H20V12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CircleIcon({ children }: { children: ReactNode }) {
  return <span className="circle-icon">{children}</span>;
}

export default function PressureDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [role, setRole] = useState<Role>("batters");
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadData = useCallback(async () => {
    try {
      setData(await requestDashboardData());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load rankings.");
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshData = () => {
    setLoading(true);
    setError("");
    void loadData();
  };

  useEffect(() => {
    let active = true;
    requestDashboardData()
      .then((result) => {
        if (active) setData(result);
      })
      .catch((caught: unknown) => {
        if (active) setError(caught instanceof Error ? caught.message : "Could not load rankings.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const allRows = data?.[role] ?? EMPTY_ROWS;
  const visibleRows = useMemo(() => {
    if (!search.trim()) return allRows.slice(0, 10);
    const needle = search.toLocaleLowerCase();
    return allRows.filter((row) => playerLabel(row, role, allRows).toLocaleLowerCase().includes(needle)).slice(0, 50);
  }, [allRows, role, search]);

  const selected = allRows.find((row) => identity(row, role) === selectedId) ?? allRows[0];
  const topFive = allRows.slice(0, 5);
  const audit = data?.audit ?? [];
  const batterView = role === "batters";
  const matches = auditValue(audit, "scoped_matches");
  const deliveryCount = auditValue(audit, "ball_by_ball_rows");

  const selectRole = (nextRole: Role) => {
    setRole(nextRole);
    setSelectedId("");
    setSearch("");
  };

  const selectedMetrics = batterView
    ? [
        { label: "Career batting average", value: value(selected, "batting_average").toFixed(1), suffix: "runs / dismissal" },
        { label: "Boundary share of runs", value: formatPercent(value(selected, "boundary_run_share"), true), suffix: "from 4s + 6s" },
        { label: "Eligible innings at SR ≥150", value: formatPercent(value(selected, "sr150_consistency"), true), suffix: `${formatNumber(value(selected, "sr_eligible_innings"))} innings` },
        { label: "Overall strike rate", value: value(selected, "overall_strike_rate").toFixed(1), suffix: "runs / 100 balls" },
      ]
    : [
        { label: "Pressure economy", value: value(selected, "pressure_economy").toFixed(2), suffix: "runs / over" },
        { label: "Pressure wickets", value: formatNumber(value(selected, "pressure_wickets")), suffix: "wickets" },
        { label: "Pressure dot-ball rate", value: formatPercent(value(selected, "pressure_dot_percentage")), suffix: `${formatNumber(value(selected, "pressure_balls"))} legal balls` },
        { label: "Wides / no-balls avoided / 100", value: value(selected, "extras_avoided_per100_shrunk").toFixed(1), suffix: "pressure balls" },
      ];

  return (
    <main className="dashboard-shell flex min-h-screen bg-rose-50">
      <aside className="sidebar flex min-h-screen flex-col bg-rose-100">
        <div className="brand-lockup">
          <BrandMark />
          <div>
            <span className="brand-name">PRESSURE<span>/</span>PLAY</span>
            <span className="brand-caption">CRICKET PERFORMANCE</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Dashboard sections">
          <button className="nav-item nav-current" type="button" onClick={() => selectRole("batters")}>
            <span className="nav-glyph">◫</span><span>Overview</span><span className="nav-dot" />
          </button>
          <button className={`nav-item ${batterView ? "nav-selected" : ""}`} type="button" onClick={() => selectRole("batters")}>
            <span className="nav-glyph">↗</span><span>Batting</span>
          </button>
          <button className={`nav-item ${!batterView ? "nav-selected" : ""}`} type="button" onClick={() => selectRole("bowlers")}>
            <span className="nav-glyph">⌁</span><span>Bowling</span>
          </button>
        </nav>

        <div className="sidebar-spacer" />
        <div className="sidebar-model">
          <div className="model-status"><span className="status-led" /> MODEL ACTIVE</div>
          <p>Context adjusted<br />pressure index</p>
          <div className="sidebar-model-foot"><span>HGB + RF benchmark</span><span>↗</span></div>
        </div>
        <div className="sidebar-footer"><span className="avatar">SA</span><span><b>Sports analytics</b><small>Performance workspace</small></span><span className="more-dots">···</span></div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><i>/</i><b>Pressure overview</b></div>
          <div className="topbar-right">
            <span className="data-window"><span className="status-led" /> T20 competitions <span className="data-divider">·</span> 2022 onward</span>
            <button className="icon-button" type="button" aria-label="Refresh rankings" onClick={refreshData} disabled={loading}>
              <svg className={loading ? "refresh-spin" : ""} viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14.9-3M4 4v4h4M4 13a8 8 0 0 0 14.9 3M20 20v-4h-4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            <div className="top-avatar">SA</div>
          </div>
        </header>

        <div className="dashboard-content mx-auto w-full">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> PERFORMANCE INTELLIGENCE</div>
              <h1>Pressure <em>changes</em> everything.</h1>
              <p>See who delivers when the game tightens. Context adjusted rankings from 2022 onward.</p>
            </div>
            <div className="heading-action">
              <span className="last-updated">DATASET COVERAGE</span>
              <span className="updated-value">{data ? `${formatNumber(matches)} matches analyzed` : "Loading data"}</span>
            </div>
          </section>

          {error ? (
            <section className="error-panel" role="alert">
              <div className="error-icon">!</div>
              <div><h2>Rankings could not be loaded</h2><p>{error}</p></div>
              <button type="button" className="button-secondary" onClick={refreshData}>Try again</button>
            </section>
          ) : (
            <>
              <section className="kpi-grid grid grid-cols-2 gap-3" aria-label="Dataset summary">
                <article className="kpi-card kpi-primary">
                  <div className="kpi-top"><span>DELIVERIES ANALYZED</span><CircleIcon><TrendIcon /></CircleIcon></div>
                  <div className="kpi-value">{loading ? "—" : formatNumber(deliveryCount)}<small>balls</small></div>
                  <div className="kpi-foot"><span className="kpi-accent">●</span> Match grouped model validation</div>
                </article>
                <article className="kpi-card">
                  <div className="kpi-top"><span>BATTERS RANKED</span><CircleIcon><span>↗</span></CircleIcon></div>
                  <div className="kpi-value">{loading ? "—" : formatNumber(data?.batters.length ?? 0)}<small>players</small></div>
                  <div className="kpi-foot">Full observed player pool</div>
                </article>
                <article className="kpi-card">
                  <div className="kpi-top"><span>BOWLERS RANKED</span><CircleIcon><span>⌁</span></CircleIcon></div>
                  <div className="kpi-value">{loading ? "—" : formatNumber(data?.bowlers.length ?? 0)}<small>players</small></div>
                  <div className="kpi-foot">Full observed player pool</div>
                </article>
                <article className="kpi-card kpi-pressure">
                  <div className="kpi-top"><span>CHASE PRESSURE</span><CircleIcon><span>!</span></CircleIcon></div>
                  <div className="kpi-value">16<small>RPO</small></div>
                  <div className="kpi-foot">Required rate trigger · second innings</div>
                </article>
              </section>

              <section className="insight-grid">
                <article className="panel leader-panel">
                  <div className="panel-heading">
                    <div><span className="section-kicker">THE PRESSURE BOARD</span><h2>Leaders in the moment</h2></div>
                    <div className="role-switch" role="tablist" aria-label="Player role">
                      <button type="button" role="tab" aria-selected={batterView} className={batterView ? "active" : ""} onClick={() => selectRole("batters")}>Batters</button>
                      <button type="button" role="tab" aria-selected={!batterView} className={!batterView ? "active" : ""} onClick={() => selectRole("bowlers")}>Bowlers</button>
                    </div>
                  </div>
                  <p className="panel-subtitle">Top five by context adjusted pressure score</p>
                  {loading ? <div className="loading-bars"><i /><i /><i /><i /><i /></div> : (
                    <div className="leader-chart">
                      {topFive.map((row, index) => {
                        const score = value(row, "score");
                        const selectedRow = identity(row, role) === identity(selected, role);
                        return (
                          <button className={`leader-row ${selectedRow ? "leader-active" : ""}`} key={identity(row, role)} type="button" onClick={() => setSelectedId(identity(row, role))}>
                            <span className="leader-rank">{String(index + 1).padStart(2, "0")}</span>
                            <span className="leader-main"><span className="leader-name">{playerLabel(row, role, allRows)}</span><span className="leader-track"><i style={{ width: `${Math.max(4, score * 100)}%` }} /></span></span>
                            <span className="leader-score">{(score * 100).toFixed(1)}<small> / 100</small></span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                  <div className="chart-foot"><span><i className="legend-dot" /> Pressure score</span><span>Reliability adjusted · 0–100</span></div>
                </article>

                <article className="panel signals-panel">
                  <div className="panel-heading signals-heading"><div><span className="section-kicker">HOW PRESSURE IS DEFINED</span><h2>Game state signals</h2></div><span className="signal-icon">⌁</span></div>
                  <div className="signal-list">
                    <div className="signal-item"><span className="signal-number">01</span><div><b>Final five overs</b><small>Last five overs of the innings</small></div><span className="signal-chip">PHASE</span></div>
                    <div className="signal-item"><span className="signal-number">02</span><div><b>Demanding chase</b><small>Required rate ≥16 runs per over</small></div><span className="signal-chip">CHASE</span></div>
                    <div className="signal-item"><span className="signal-number">03</span><div><b>Wicket pressure</b><small>Five or more wickets lost</small></div><span className="signal-chip">RISK</span></div>
                    <div className="signal-item"><span className="signal-number">04</span><div><b>Semi-final / final</b><small>Also includes qualifiers and eliminators</small></div><span className="signal-chip">STAGE</span></div>
                    <div className="signal-item"><span className="signal-number">05</span><div><b>First-innings recovery</b><small>First half · 2+ wickets lost · batting score</small></div><span className="signal-chip">BATTERS</span></div>
                    <div className="signal-item"><span className="signal-number">06</span><div><b>Powerplay collapse</b><small>Overs 1–6 · 2+ wickets lost · batting score</small></div><span className="signal-chip">BATTERS</span></div>
                    <div className="signal-item"><span className="signal-number">07</span><div><b>Middle-order response</b><small>Batting positions 4–6 in pressure or after 2+ wickets</small></div><span className="signal-chip">BATTERS</span></div>
                    <div className="signal-item"><span className="signal-number">08</span><div><b>Opener consistency</b><small>30+ runs · SR 160+ · 50%+ runs from boundaries</small></div><span className="signal-chip">OPENERS</span></div>
                    <div className="signal-item"><span className="signal-number">09</span><div><b>Middle-order spin handling</b><small>Strike rate and runs above context vs spin</small></div><span className="signal-chip">BATTERS</span></div>
                    <div className="signal-item"><span className="signal-number">10</span><div><b>Powerplay partnership break</b><small>Overs 1–6 · stand 30+ · striker 20+ from 12+ balls</small></div><span className="signal-chip">BOWLERS</span></div>
                    <div className="signal-item"><span className="signal-number">11</span><div><b>Death-over settled-batter pressure</b><small>Last five overs · 6+ wickets in hand · stand 30+ · striker 20+ from 12+ balls</small></div><span className="signal-chip">BOWLERS</span></div>
                  </div>
                  <div className="signals-foot"><span className="signals-foot-icon">i</span><span>Each delivery can qualify through one or more signals.</span></div>
                </article>
              </section>

              <section className="ranking-layout">
                <article className="panel ranking-panel">
                  <div className="panel-heading ranking-heading">
                    <div><span className="section-kicker">FULL PLAYER POOL</span><h2>{batterView ? "Batter rankings" : "Bowler rankings"}</h2></div>
                    <div className="table-actions">
                      <label className="search-box">
                        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8" stroke="currentColor" strokeWidth="1.7" /><path d="m16 16 4.2 4.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>
                        <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search all players" aria-label="Search all players" />
                        {search && <button type="button" onClick={() => setSearch("")} aria-label="Clear search">×</button>}
                      </label>
                    </div>
                  </div>
                  <div className="table-caption"><span>{search ? `${visibleRows.length} matches for “${search}”` : "Top 10 · all player records searchable"}</span><span>Rank score <span className="caption-info">i</span></span></div>
                  <div className="table-scroll">
                    <table>
                      <thead><tr>
                        <th className="rank-col">RANK</th><th>PLAYER</th>
                        {batterView ? <><th>PRESSURE BALLS</th><th>PRESSURE SR</th><th>AVG</th><th>BOUNDARY RUN SHARE</th><th>INNINGS · SR ≥150</th><th>ROLE STANDARD</th><th>OPENER 30+ / 160+ / BOUNDARIES</th><th>POWERPLAY COLLAPSE +/100</th><th>MIDDLE-ORDER PRESSURE +/100</th><th>MIDDLE-ORDER SPIN SR</th><th>SPIN RUNS ABOVE EXPECTED / 100</th><th>RECOVERY BALLS</th><th>RECOVERY +/100</th><th>LAST 5 OVERS · 160+</th></> : <><th>PRESSURE WKTS</th><th>ECONOMY</th><th>DOT BALLS</th><th>WIDES / NO-BALLS / 100</th><th>PP SET BATTERS DISMISSED / OPPORTUNITIES</th><th>PP BREAKS ABOVE EXPECTED / 100</th><th>DEATH SET WKTS / BALLS</th><th>DEATH SET ECONOMY</th><th>DEATH SET WKTS ABOVE EXPECTED / 100</th><th>DEATH SET RUNS SAVED / 100</th></>}
                        <th className="score-col">SCORE</th>
                      </tr></thead>
                      <tbody>
                        {visibleRows.map((row) => {
                          const player = playerLabel(row, role, allRows);
                          const playerId = identity(row, role);
                          const isSelected = playerId === identity(selected, role);
                          return (
                            <tr className={isSelected ? "selected-table-row" : ""} key={playerId} onClick={() => setSelectedId(playerId)}>
                              <td className="rank-col"><span className={`rank-number ${Number(row.rank) <= 3 ? "medal-rank" : ""}`}>{String(value(row, "rank")).padStart(2, "0")}</span></td>
                              <td><button type="button" className="player-cell" onClick={() => setSelectedId(playerId)}><span className="player-avatar">{label(row, role).split(/\s+/).map((part) => part[0]).slice(0, 2).join("")}</span><b>{player}</b></button></td>
                              {batterView ? <>
                                <td>{formatNumber(value(row, "pressure_balls"))}</td>
                                <td><span className="metric-emphasis">{value(row, "pressure_strike_rate").toFixed(1)}</span></td>
                                <td>{value(row, "batting_average").toFixed(1)}</td>
                                <td>{formatPercent(value(row, "boundary_run_share"), true)}</td>
                                <td>{formatPercent(value(row, "sr150_consistency"), true)}</td>
                                <td><span className="rate-pill">{battingRoleStatus(row)}</span></td>
                                <td>{formatPercent(value(row, "opener_consistency_rate"), true)} <small>({formatNumber(value(row, "opener_qualifying_innings"))}/{formatNumber(value(row, "opener_innings"))})</small></td>
                                <td>{value(row, "powerplay_collapse_rae_per100_shrunk").toFixed(1)} <small>({formatNumber(value(row, "powerplay_collapse_balls"))})</small></td>
                                <td>{value(row, "middle_order_pressure_rae_per100_shrunk").toFixed(1)} <small>({formatNumber(value(row, "middle_order_pressure_balls"))})</small></td>
                                <td>{value(row, "middle_order_spin_strike_rate").toFixed(1)} <small>({formatNumber(value(row, "middle_order_spin_balls"))} balls)</small></td>
                                <td>{value(row, "middle_order_spin_rae_per100_shrunk").toFixed(1)}</td>
                                <td>{formatNumber(value(row, "early_recovery_balls"))}</td>
                                <td>{value(row, "early_recovery_rae_per100_shrunk").toFixed(1)}</td>
                                <td><span className="rate-pill">{formatPercent(value(row, "last5_sr_160_rate"), true)}</span></td>
                              </> : <>
                                <td>{formatNumber(value(row, "pressure_wickets"))}</td>
                                <td>{value(row, "pressure_economy").toFixed(2)}</td>
                                <td><span className="rate-pill">{formatPercent(value(row, "pressure_dot_percentage"))}</span></td>
                                <td>{value(row, "extras_avoided_per100_shrunk").toFixed(2)}</td>
                                <td>{formatNumber(value(row, "powerplay_set_partnership_breaks"))} / {formatNumber(value(row, "powerplay_set_partnership_opportunities"))}</td>
                                <td>{value(row, "powerplay_set_partnership_breaks_above_expected_per100_shrunk").toFixed(2)}</td>
                                <td>{formatNumber(value(row, "death_settled_batter_wickets"))} / {formatNumber(value(row, "death_settled_batter_opportunities"))}</td>
                                <td>{value(row, "death_settled_batter_opportunities") > 0 ? value(row, "death_settled_batter_economy").toFixed(2) : "—"}</td>
                                <td>{value(row, "death_settled_batter_opportunities") >= 6 ? value(row, "death_settled_batter_wickets_above_expected_per100_shrunk").toFixed(2) : "—"}</td>
                                <td>{value(row, "death_settled_batter_opportunities") >= 6 ? value(row, "death_settled_batter_runs_saved_per100_shrunk").toFixed(2) : "—"}</td>
                              </>}
                              <td className="score-col"><span className="score-badge">{(value(row, "score") * 100).toFixed(1)}</span></td>
                            </tr>
                          );
                        })}
                        {!loading && visibleRows.length === 0 && <tr><td className="empty-cell" colSpan={batterView ? 8 : 7}>No players match that search.</td></tr>}
                      </tbody>
                    </table>
                  </div>
                  <div className="ranking-foot"><span>Showing {search ? `search results` : `top 10 of ${formatNumber(allRows.length)}`}</span><span>Click any player to inspect their pressure profile <span className="arrow-right">↗</span></span></div>
                </article>

                <aside className="panel player-detail">
                  <div className="detail-topline"><span className="section-kicker">PLAYER SNAPSHOT</span><span className="snapshot-live"><i /> RANKED</span></div>
                  {selected ? <>
                    <div className="detail-identity">
                      <div className="detail-avatar">{label(selected, role).split(/\s+/).map((part) => part[0]).slice(0, 2).join("")}</div>
                      <div><span className="detail-rank">RANK #{String(value(selected, "rank")).padStart(2, "0")}</span><h3>{playerLabel(selected, role, allRows)}</h3><span className="detail-role">{batterView ? battingRoleStatus(selected) : "BOWLER · T20 FORMATS"}</span></div>
                    </div>
                    <div className="detail-score-row"><span>Pressure score</span><b>{(value(selected, "score") * 100).toFixed(1)}<small> / 100</small></b></div>
                    <div className="score-meter"><i style={{ width: `${Math.max(2, value(selected, "score") * 100)}%` }} /></div>
                    <div className="detail-metrics">
                      {selectedMetrics.map((metric) => <div className="detail-metric" key={metric.label}><span>{metric.label}</span><b>{metric.value}<small>{metric.suffix}</small></b></div>)}
                    </div>
                    <div className="detail-note"><span className="detail-note-mark">✳</span><p>{batterView ? "Batting score includes above-expected runs, recovery after early first-innings wickets, and 160+ strike rate success in the final five overs." : "Bowling score rewards runs saved, wickets, discipline, and dots above context expectation, plus powerplay breaks and death-over wicket taking with good economy against set batters."}</p></div>
                  </> : <div className="detail-empty">Player profile will appear here.</div>}
                </aside>
              </section>

              <footer className="dashboard-footer">
                <div><BrandMark /><span>PRESSURE<span>/</span>PLAY</span></div>
                <span>Context only · match grouped validation · recency weighted</span>
                <span>Data: {formatNumber(matches)} matches <i>·</i> {formatNumber(deliveryCount)} deliveries</span>
              </footer>
            </>
          )}
        </div>
      </section>
    </main>
  );
}
