"use client";
import { useState, useEffect, useCallback } from "react";
import type { GraphState, RankEntry, PairwisePrediction, PerGameStats } from "@/lib/graph-engine";

type Tab = "ranking" | "log" | "matchup" | "history" | "players";

const EMPTY_STATS = (): PerGameStats => ({ kills: 0, deaths: 0, assists: 0, headshots: 0 });

function StatInput({ label, stats, onChange }: {
  label: string; stats: PerGameStats; onChange: (s: PerGameStats) => void;
}) {
  const fields: (keyof PerGameStats)[] = ["kills", "deaths", "assists", "headshots"];
  return (
    <div style={{ flex: 1 }}>
      <div className="section-label" style={{ marginBottom: 8 }}>{label}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {fields.map(f => (
          <div key={f}>
            <div style={{ fontSize: "0.58rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", letterSpacing: "0.1em", marginBottom: 2 }}>{f.toUpperCase()}</div>
            <input className="input" type="number" min="0" max="99" value={stats[f]}
              onChange={e => onChange({ ...stats, [f]: parseInt(e.target.value) || 0 })}
              style={{ padding: "4px 8px", fontSize: "0.9rem" }} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("ranking");
  const [state, setState] = useState<GraphState>({ players: {}, games: [] });
  const [ranking, setRanking] = useState<RankEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [rankLoading, setRankLoading] = useState(false);
  const [error, setError] = useState("");

  // Log tab
  const [logWinner, setLogWinner] = useState("");
  const [logLoser, setLogLoser] = useState("");
  const [winStats, setWinStats] = useState<PerGameStats>(EMPTY_STATS());
  const [loseStats, setLoseStats] = useState<PerGameStats>(EMPTY_STATS());
  const [logging, setLogging] = useState(false);
  const [logMsg, setLogMsg] = useState("");

  // Matchup tab
  const [predA, setPredA] = useState("");
  const [predB, setPredB] = useState("");
  const [prediction, setPrediction] = useState<PairwisePrediction | null>(null);
  const [predicting, setPredicting] = useState(false);

  // Players tab
  const [newName, setNewName] = useState("");
  const [addingPlayer, setAddingPlayer] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      const s = await res.json();
      if (!res.ok) { setError(s.error ?? "Failed to load state"); return null; }
      setState(s);
      return s;
    } catch { setError("Connection failed"); return null; }
    finally { setLoading(false); }
  }, []);

  const fetchRanking = useCallback(async () => {
    setRankLoading(true);
    try {
      const res = await fetch("/api/predict", { cache: "no-store" });
      setRanking(await res.json());
    } catch { setError("Ranking failed"); }
    finally { setRankLoading(false); }
  }, []);

  useEffect(() => {
    refresh().then(s => { if (s && Object.keys(s.players).length > 0) fetchRanking(); });
  }, [refresh, fetchRanking]);

  const players = Object.values(state.players);

  async function addPlayer(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    setAddingPlayer(true);
    try {
      const res = await fetch("/api/players", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: newName.trim() }),
      });
      if (!res.ok) { const d = await res.json(); setError(d.error); }
      else { const s = await res.json(); setState(s); setNewName(""); }
    } finally { setAddingPlayer(false); }
  }

  async function logGame(e: React.FormEvent) {
    e.preventDefault();
    if (!logWinner || !logLoser) return;
    setLogging(true); setLogMsg("");
    try {
      const res = await fetch("/api/matches", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ winner_id: logWinner, loser_id: logLoser, winner_stats: winStats, loser_stats: loseStats }),
      });
      if (!res.ok) { const d = await res.json(); setError(d.error); }
      else {
        const s = await res.json();
        setLogMsg(`✓ ${state.players[logWinner]?.display_name} def. ${state.players[logLoser]?.display_name}`);
        setLogWinner(""); setLogLoser("");
        setWinStats(EMPTY_STATS()); setLoseStats(EMPTY_STATS());
        setState(s);
        fetchRanking();
      }
    } finally { setLogging(false); }
  }

  async function getPrediction() {
    if (!predA || !predB || predA === predB) return;
    setPredicting(true); setPrediction(null);
    try {
      const res = await fetch(`/api/predict?a=${predA}&b=${predB}`);
      setPrediction(await res.json());
    } finally { setPredicting(false); }
  }

  function pct(n: number, dec = 0) {
    return `${(n * 100).toFixed(dec)}%`;
  }
  function fmt(n: number, dec = 2) { return n.toFixed(dec); }
  function confLabel(c: number) { return c < 0.25 ? "LOW" : c < 0.6 ? "MED" : "HIGH"; }
  function confColor(c: number) { return c < 0.25 ? "var(--lose)" : c < 0.6 ? "var(--neutral)" : "var(--win)"; }

  const recentGames = [...state.games].reverse().slice(0, 30);

  return (
    <div style={{ minHeight: "100vh", padding: "24px", maxWidth: 900, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <div className="live-indicator" />
          <span className="font-mono" style={{ fontSize: "0.6rem", color: "var(--accent)", letterSpacing: "0.2em" }}>
            GRAPHELO // R6 CUSTOM 1v1
          </span>
        </div>
        <div className="gradient-bar" style={{ marginBottom: 8 }} />
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1 className="font-display" style={{ fontSize: "2rem", fontWeight: 700, color: "var(--text-bright)", letterSpacing: "0.05em" }}>
            GRAPH<span style={{ color: "var(--accent)" }}>ELO</span>
          </h1>
          <span className="font-mono" style={{ fontSize: "0.65rem", color: "var(--text-dim)" }}>
            {players.length} PLAYERS · {state.games.length} GAMES
          </span>
        </div>
      </div>

      {error && (
        <div style={{ marginBottom: 14, padding: "7px 12px", border: "1px solid var(--lose)", color: "var(--lose)", fontSize: "0.78rem", fontFamily: "Share Tech Mono", display: "flex", justifyContent: "space-between" }}>
          <span>⚠ {error}</span>
          <button onClick={() => setError("")} style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer" }}>×</button>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--border)", marginBottom: 20 }}>
        {(["ranking", "log", "matchup", "history", "players"] as Tab[]).map(t => (
          <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {loading && <div className="font-mono" style={{ color: "var(--text-dim)", fontSize: "0.8rem" }}>LOADING...</div>}

      {/* ── RANKING ── */}
      {!loading && tab === "ranking" && (
        <div className="fade-in">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <div className="section-label">ROUND ROBIN RANKING</div>
            <span className="font-mono" style={{ fontSize: "0.6rem", color: "var(--text-dim)" }}>1000 RR / PAIRING · PATHS WEIGHTED 1/L · EXACT E[WINS]</span>
            <button className="btn" style={{ marginLeft: "auto", padding: "3px 10px", fontSize: "0.7rem" }} onClick={fetchRanking} disabled={rankLoading}>
              {rankLoading ? "CALCULATING..." : "↺ RECALC"}
            </button>
          </div>

          {ranking.length === 0 ? (
            <div style={{ color: "var(--text-dim)", fontSize: "0.85rem" }}>
              {players.length < 2 ? "Add at least 2 players and log a game." : "Log some games to generate a ranking."}
            </div>
          ) : (
            <>
              {/* Table header */}
              <div style={{ display: "grid", gridTemplateColumns: "28px 1fr 80px 60px 60px 60px 60px 60px", gap: 10, padding: "5px 12px", marginBottom: 4 }}>
                {["#", "PLAYER", "WIN%", "KD", "KDA", "HS%", "KPR", "GAMES"].map((h, i) => (
                  <span key={h} className="section-label" style={{ textAlign: i > 1 ? "right" : "left" }}>{h}</span>
                ))}
              </div>

              {ranking.map((r, i) => {
                const sv = r.stat_vec;
                const barW = Math.max(4, Math.round(r.tournament_win_pct * 100));
                const isTop = i === 0;
                return (
                  <div key={r.player_id} className="panel corner-tl" style={{
                    display: "grid",
                    gridTemplateColumns: "28px 1fr 80px 60px 60px 60px 60px 60px",
                    gap: 10, padding: "11px 12px", alignItems: "center",
                    marginBottom: 1,
                    borderColor: isTop ? "var(--accent)" : undefined,
                  }}>
                    <span className="rank-number" style={{ color: isTop ? "var(--accent)" : undefined, fontWeight: isTop ? 700 : undefined }}>
                      {i + 1}
                    </span>
                    <div>
                      <div className="player-name" style={{ color: isTop ? "var(--accent)" : undefined }}>{r.display_name}</div>
                      <div className="rating-bar" style={{ marginTop: 4, width: "85%" }}>
                        <div className="rating-bar-fill" style={{ width: `${barW}%` }} />
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <span className="rating-value" style={{ fontSize: "0.95rem" }}>{pct(r.tournament_win_pct, 1)}</span>
                    </div>
                    <div style={{ textAlign: "right" }} className="winrate">{fmt(sv.kd)}</div>
                    <div style={{ textAlign: "right" }} className="winrate">{fmt(sv.kda)}</div>
                    <div style={{ textAlign: "right" }} className="winrate">{pct(sv.hs_pct, 1)}</div>
                    <div style={{ textAlign: "right" }} className="winrate">{fmt(sv.kpr, 1)}</div>
                    <div style={{ textAlign: "right", color: "var(--text-dim)" }} className="winrate">{sv.games_played}</div>
                  </div>
                );
              })}

              {/* Matchup matrix */}
              {ranking.length >= 2 && (
                <div style={{ marginTop: 28 }}>
                  <div className="section-label" style={{ marginBottom: 10 }}>PAIRWISE WIN PROBABILITY MATRIX (row beats col)</div>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ borderCollapse: "collapse", fontSize: "0.72rem", fontFamily: "Share Tech Mono" }}>
                      <thead>
                        <tr>
                          <td style={{ padding: "4px 10px", color: "var(--text-dim)" }}></td>
                          {ranking.map(r => (
                            <td key={r.player_id} style={{ padding: "4px 8px", color: "var(--text-dim)", textAlign: "center", maxWidth: 70, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {r.display_name.slice(0, 6)}
                            </td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {ranking.map(rowP => (
                          <tr key={rowP.player_id}>
                            <td style={{ padding: "4px 10px", color: "var(--text-dim)", whiteSpace: "nowrap" }}>{rowP.display_name.slice(0, 8)}</td>
                            {ranking.map(colP => {
                              if (rowP.player_id === colP.player_id) {
                                return <td key={colP.player_id} style={{ padding: "4px 8px", textAlign: "center", background: "var(--surface2)", color: "var(--text-dim)" }}>—</td>;
                              }
                              const p = rowP.matchup_table[colP.player_id] ?? 0.5;
                              const color = p > 0.6 ? "var(--win)" : p < 0.4 ? "var(--lose)" : "var(--neutral)";
                              return (
                                <td key={colP.player_id} style={{ padding: "4px 8px", textAlign: "center", color }}>
                                  {pct(p, 0)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── LOG ── */}
      {tab === "log" && (
        <div className="fade-in" style={{ maxWidth: 580 }}>
          <div className="section-label" style={{ marginBottom: 14 }}>LOG 1v1 GAME</div>
          {players.length < 2 && (
            <div style={{ color: "var(--neutral)", fontSize: "0.82rem", marginBottom: 14, fontFamily: "Share Tech Mono" }}>⚠ Add at least 2 players first</div>
          )}
          <form onSubmit={logGame}>
            {/* Winner / Loser selectors */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 36px 1fr", gap: 12, alignItems: "end", marginBottom: 16 }}>
              <div>
                <div className="section-label" style={{ marginBottom: 6 }}>WINNER</div>
                <select className="input" value={logWinner} onChange={e => setLogWinner(e.target.value)}>
                  <option value="">— select —</option>
                  {players.map(p => <option key={p.id} value={p.id}>{p.display_name}</option>)}
                </select>
              </div>
              <div style={{ textAlign: "center", fontFamily: "Rajdhani", fontWeight: 700, color: "var(--text-dim)", paddingBottom: 10 }}>VS</div>
              <div>
                <div className="section-label" style={{ marginBottom: 6 }}>LOSER</div>
                <select className="input" value={logLoser} onChange={e => setLogLoser(e.target.value)}>
                  <option value="">— select —</option>
                  {players.filter(p => p.id !== logWinner).map(p => <option key={p.id} value={p.id}>{p.display_name}</option>)}
                </select>
              </div>
            </div>

            {/* Stats inputs */}
            <div className="panel" style={{ padding: 16, marginBottom: 14 }}>
              <div style={{ display: "flex", gap: 20 }}>
                <StatInput label={logWinner ? `${state.players[logWinner]?.display_name ?? "WINNER"} STATS` : "WINNER STATS"} stats={winStats} onChange={setWinStats} />
                <div style={{ width: 1, background: "var(--border)", flexShrink: 0 }} />
                <StatInput label={logLoser ? `${state.players[logLoser]?.display_name ?? "LOSER"} STATS` : "LOSER STATS"} stats={loseStats} onChange={setLoseStats} />
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button type="submit" className="btn btn-primary" disabled={logging || !logWinner || !logLoser || players.length < 2}>
                {logging ? "LOGGING..." : "LOG GAME"}
              </button>
              {logMsg && <span className="font-mono" style={{ color: "var(--win)", fontSize: "0.78rem" }}>{logMsg}</span>}
            </div>
          </form>
        </div>
      )}

      {/* ── MATCHUP ── */}
      {tab === "matchup" && (
        <div className="fade-in" style={{ maxWidth: 520 }}>
          <div className="section-label" style={{ marginBottom: 14 }}>PAIRWISE PREDICTION</div>
          <div className="panel" style={{ padding: 18, marginBottom: 14 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 36px 1fr", gap: 12, alignItems: "end", marginBottom: 14 }}>
              <div>
                <div className="section-label" style={{ marginBottom: 6 }}>PLAYER A</div>
                <select className="input" value={predA} onChange={e => { setPredA(e.target.value); setPrediction(null); }}>
                  <option value="">— select —</option>
                  {players.map(p => <option key={p.id} value={p.id}>{p.display_name}</option>)}
                </select>
              </div>
              <div style={{ textAlign: "center", fontFamily: "Rajdhani", fontWeight: 700, color: "var(--text-dim)", paddingBottom: 10 }}>VS</div>
              <div>
                <div className="section-label" style={{ marginBottom: 6 }}>PLAYER B</div>
                <select className="input" value={predB} onChange={e => { setPredB(e.target.value); setPrediction(null); }}>
                  <option value="">— select —</option>
                  {players.filter(p => p.id !== predA).map(p => <option key={p.id} value={p.id}>{p.display_name}</option>)}
                </select>
              </div>
            </div>
            <button className="btn btn-primary" onClick={getPrediction} disabled={!predA || !predB || predicting} style={{ width: "100%" }}>
              {predicting ? "TRAVERSING GRAPH..." : "PREDICT"}
            </button>
          </div>

          {prediction && (() => {
            const aName = state.players[predA]?.display_name ?? "A";
            const bName = state.players[predB]?.display_name ?? "B";
            const pA = prediction.p_a_wins;
            const pB = 1 - pA;
            return (
              <div className="fade-in">
                <div className="panel panel-accent2" style={{ padding: 18 }}>
                  {/* Win probability bar */}
                  <div style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <div>
                        <span className="player-name">{aName}</span>
                        <span className="rating-value" style={{ fontSize: "1.6rem", marginLeft: 12 }}>{pct(pA, 1)}</span>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <span className="rating-value" style={{ fontSize: "1.6rem", color: pB > pA ? "var(--accent)" : "var(--lose)", marginRight: 12 }}>{pct(pB, 1)}</span>
                        <span className="player-name">{bName}</span>
                      </div>
                    </div>
                    {/* Split bar */}
                    <div style={{ height: 6, background: "var(--lose)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: pct(pA), background: "var(--accent)", transition: "width 0.5s" }} />
                    </div>
                  </div>

                  {/* Evidence breakdown */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                    {[
                      { label: "DIRECT GAMES", val: prediction.direct_games.toString() },
                      { label: "GRAPH PATHS", val: prediction.paths_used.toString() },
                      { label: "EVIDENCE", val: fmt(prediction.evidence_mass, 2) },
                      { label: "CONFIDENCE", val: confLabel(prediction.confidence), color: confColor(prediction.confidence) },
                    ].map(({ label, val, color }) => (
                      <div key={label} style={{ padding: "8px 10px", border: "1px solid var(--border)" }}>
                        <div className="section-label" style={{ marginBottom: 4 }}>{label}</div>
                        <div className="font-mono" style={{ fontSize: "0.95rem", color: color ?? "var(--text-bright)" }}>{val}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── HISTORY ── */}
      {tab === "history" && (
        <div className="fade-in">
          <div className="section-label" style={{ marginBottom: 12 }}>GAME HISTORY ({state.games.length})</div>
          {recentGames.length === 0 && <div style={{ color: "var(--text-dim)", fontSize: "0.85rem" }}>No games logged yet.</div>}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {recentGames.map((g, i) => {
              const w = state.players[g.winner_id];
              const l = state.players[g.loser_id];
              const ws = g.winner_stats;
              const ls = g.loser_stats;
              const wKD = ws.deaths > 0 ? (ws.kills / ws.deaths).toFixed(2) : ws.kills.toString();
              const lKD = ls.deaths > 0 ? (ls.kills / ls.deaths).toFixed(2) : ls.kills.toString();
              const wHS = ws.kills > 0 ? `${Math.round(ws.headshots / ws.kills * 100)}%` : "—";
              const lHS = ls.kills > 0 ? `${Math.round(ls.headshots / ls.kills * 100)}%` : "—";
              return (
                <div key={g.id} className="panel" style={{ padding: "10px 14px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
                    {/* Winner side */}
                    <div style={{ flex: 1 }}>
                      <span className="player-name" style={{ color: "var(--win)" }}>{w?.display_name ?? g.winner_id}</span>
                      <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--text-dim)", marginTop: 2 }}>
                        {ws.kills}K {ws.deaths}D {ws.assists}A · KD {wKD} · HS {wHS}
                      </div>
                    </div>
                    <div style={{ padding: "0 14px", textAlign: "center" }}>
                      <div className="font-mono" style={{ fontSize: "0.6rem", color: "var(--accent)" }}>DEF</div>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono" }}>
                        {new Date(g.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                    {/* Loser side */}
                    <div style={{ flex: 1, textAlign: "right" }}>
                      <span className="player-name" style={{ color: "var(--lose)" }}>{l?.display_name ?? g.loser_id}</span>
                      <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--text-dim)", marginTop: 2 }}>
                        HS {lHS} · KD {lKD} · {ls.assists}A {ls.deaths}D {ls.kills}K
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── PLAYERS ── */}
      {tab === "players" && (
        <div className="fade-in" style={{ maxWidth: 440 }}>
          <div className="section-label" style={{ marginBottom: 12 }}>ADD PLAYER</div>
          <form onSubmit={addPlayer} className="panel" style={{ padding: 18, marginBottom: 20, display: "flex", gap: 10 }}>
            <input className="input" value={newName} onChange={e => setNewName(e.target.value)} placeholder="Display name (e.g. grug)" style={{ flex: 1 }} />
            <button type="submit" className="btn btn-primary" disabled={addingPlayer || !newName.trim()}>
              {addingPlayer ? "..." : "ADD"}
            </button>
          </form>
          <div className="section-label" style={{ marginBottom: 8 }}>ROSTER ({players.length})</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
            {players.length === 0 && <div style={{ color: "var(--text-dim)", fontSize: "0.82rem" }}>No players yet.</div>}
            {players.map(p => {
              const r = ranking.find(x => x.player_id === p.id);
              const sv = r?.stat_vec;
              return (
                <div key={p.id} className="panel" style={{ padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div className="player-name">{p.display_name}</div>
                    {sv && sv.games_played > 0 && (
                      <div className="font-mono" style={{ fontSize: "0.62rem", color: "var(--text-dim)", marginTop: 2 }}>
                        KD {fmt(sv.kd)} · KDA {fmt(sv.kda)} · HS {pct(sv.hs_pct, 0)} · {sv.games_played} games
                      </div>
                    )}
                  </div>
                  {r && (
                    <div style={{ textAlign: "right" }}>
                      <div className="rating-value" style={{ fontSize: "0.9rem" }}>{pct(r.tournament_win_pct, 1)}</div>
                      <div style={{ fontSize: "0.6rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono" }}>TOUR WIN%</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: 16, padding: "10px 12px", border: "1px dashed var(--border)", fontSize: "0.65rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", lineHeight: 1.6 }}>
            API HOOK READY — swap computeStatVector() in lib/graph-engine.ts<br />
            with r6data.com fetch to pull ranked stats automatically.
          </div>
        </div>
      )}

      <div style={{ marginTop: 36, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
        <div className="font-mono" style={{ fontSize: "0.58rem", color: "var(--text-dim)", letterSpacing: "0.1em" }}>
          GRAPHELO v2 · GRAPH PATH WEIGHTS 1/L · τ=90d · 1000-ROUND ROBIN SIM
        </div>
      </div>
    </div>
  );
}
