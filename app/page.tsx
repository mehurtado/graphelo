"use client";
import { useState, useEffect, useCallback, useMemo } from "react";
import type { GraphState, Game, RankEntry, PairwisePrediction, PerGameStats, H2hSim, BTResult, CycleAnalysis, LooCvResult, SkillGapTrend } from "@/lib/graph-engine";
import { simulateRoundRobin, predictPairwise, computeElo, computePredictionAccuracy, computeEloHistory, computeMetaStability, computeReliability, computeEloVelocity, computeBradleyTerry, computeCycles, computeCeilingEstimate, computeNemesisProfile, computeParityIndex, computeFormRating, computeLooCvBrier, computeSkillGapTrend, computeRematchUrgency, computeUpsetAlert, kendallTauAgreement } from "@/lib/graph-engine";

type Tab = "ranking" | "log" | "matchup" | "history" | "players" | "system";

const LIGHT_VARS: Record<string, string> = {
  "--bg":         "#eef3f8",
  "--surface":    "#ffffff",
  "--surface2":   "#e2eaf2",
  "--border":     "#b8cad8",
  "--border2":    "#8aabbc",
  "--accent":     "#0076b5",
  "--accent2":    "#d44e00",
  "--accent3":    "#1a8800",
  "--text":       "#1e2f3e",
  "--text-dim":   "#567080",
  "--text-bright":"#0a1828",
  "--win":        "#1a8000",
  "--lose":       "#c01010",
  "--neutral":    "#7a5a00",
};


const EMPTY_STATS = (): PerGameStats => ({ kills: 0, deaths: 0 });


const PLAYER_COLORS = ["#f59e0b","#34d399","#f87171","#a78bfa","#38bdf8","#fb923c","#e879f9","#4ade80","#f472b6","#60a5fa"];

function EloChart({
  history, players,
}: {
  history: Record<string, Array<{ timestamp: number; elo: number }>>;
  players: Record<string, { display_name: string }>;
}) {
  const ids = Object.keys(history).filter(id => history[id].length > 0);
  if (ids.length === 0) return null;
  const allPts = ids.flatMap(id => history[id]);
  const minT = Math.min(...allPts.map(p => p.timestamp));
  const maxT = Math.max(...allPts.map(p => p.timestamp));
  const rawMin = Math.min(1000, ...allPts.map(p => p.elo));
  const rawMax = Math.max(1000, ...allPts.map(p => p.elo));
  const ePad = Math.max(15, Math.ceil((rawMax - rawMin) * 0.15));
  const minElo = rawMin - ePad, maxElo = rawMax + ePad;
  const W = 760, H = 180, PL = 44, PR = 92, PT = 10, PB = 18;
  const cw = W - PL - PR, ch = H - PT - PB;
  const xS = (t: number) => PL + (maxT === minT ? cw / 2 : ((t - minT) / (maxT - minT)) * cw);
  const yS = (e: number) => PT + ch - ((e - minElo) / (maxElo - minElo)) * ch;
  const ticks = Array.from({ length: 5 }, (_, i) => Math.round(minElo + (i / 4) * (maxElo - minElo)));
  return (
    <div>
      <div className="section-label" style={{ marginBottom: 8 }}>ELO HISTORY</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", overflow: "visible" }}>
        {ticks.map(e => (
          <g key={e}>
            <line x1={PL} y1={yS(e)} x2={W - PR} y2={yS(e)}
              stroke={e === 1000 ? "var(--border)" : "var(--surface2)"}
              strokeWidth={e === 1000 ? 0.8 : 0.4}
              strokeDasharray={e === 1000 ? "4 3" : undefined} />
            <text x={PL - 4} y={yS(e) + 3.5} fill="var(--text-dim)" fontSize={8} textAnchor="end" fontFamily="Share Tech Mono">{e}</text>
          </g>
        ))}
        {ids.map((id, ci) => {
          const pts = history[id];
          const color = PLAYER_COLORS[ci % PLAYER_COLORS.length];
          const draw = [{ timestamp: minT, elo: 1000 }, ...pts];
          const d = draw.map((p, i) => `${i === 0 ? "M" : "L"}${xS(p.timestamp).toFixed(1)},${yS(p.elo).toFixed(1)}`).join(" ");
          const last = draw[draw.length - 1];
          return (
            <g key={id}>
              <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" opacity={0.85} />
              <circle cx={xS(last.timestamp)} cy={yS(last.elo)} r={3} fill={color} />
              <text x={xS(last.timestamp) + 6} y={yS(last.elo) + 4} fill={color} fontSize={9} fontFamily="Share Tech Mono">
                {(players[id]?.display_name ?? id).slice(0, 10)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function GraphViz({ state, elo }: { state: { players: Record<string, { display_name: string }>; games: Game[] }; elo: Record<string, number> }) {
  const players = Object.values(state.players as Record<string, { display_name: string; id?: string }>);
  const ids = Object.keys(state.players);
  const n = ids.length;
  if (n < 2) return null;

  const W = 500, H = 500, cx = 250, cy = 250, R = 175, nr = 26;
  const angle = (i: number) => (2 * Math.PI * i / n) - Math.PI / 2;
  const pos: Record<string, { x: number; y: number }> = {};
  ids.forEach((id, i) => { pos[id] = { x: cx + R * Math.cos(angle(i)), y: cy + R * Math.sin(angle(i)) }; });

  const edgeMap: Record<string, Record<string, number>> = {};
  for (const id of ids) edgeMap[id] = {};
  for (const g of (state.games as Array<{ winner_id: string; loser_id: string }>) ) {
    edgeMap[g.winner_id][g.loser_id] = (edgeMap[g.winner_id][g.loser_id] ?? 0) + 1;
  }

  const edges: Array<{ from: string; to: string; total: number; dominance: number }> = [];
  const seen = new Set<string>();
  for (const a of ids) {
    for (const b of ids) {
      if (a === b) continue;
      const key = [a, b].sort().join("_");
      if (seen.has(key)) continue;
      seen.add(key);
      const aW = edgeMap[a][b] ?? 0, bW = edgeMap[b][a] ?? 0;
      const total = aW + bW;
      if (total === 0) continue;
      const dom = aW >= bW ? { from: a, to: b, dominance: aW / total } : { from: b, to: a, dominance: bW / total };
      edges.push({ ...dom, total });
    }
  }

  const connectedIds = ids.filter(id => edges.some(e => e.from === id || e.to === id));
  const hasIncoming = new Set(edges.map(e => e.to));
  const hasOutgoing = new Set(edges.map(e => e.from));
  const sources = connectedIds.filter(id => hasOutgoing.has(id) && !hasIncoming.has(id));
  const sinks   = connectedIds.filter(id => hasIncoming.has(id) && !hasOutgoing.has(id));

  return (
    <div>
      <div className="section-label" style={{ marginBottom: 8 }}>GRAPH STRUCTURE</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", maxWidth: 500, margin: "0 auto" }}>
        <defs>
          <marker id="arr-w" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 z" fill="var(--accent)" />
          </marker>
          <marker id="arr-e" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 z" fill="var(--neutral)" />
          </marker>
        </defs>
        {edges.map(({ from, to, total, dominance }) => {
          const pf = pos[from], pt = pos[to];
          const dx = pt.x - pf.x, dy = pt.y - pf.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const ux = dx / dist, uy = dy / dist;
          const x1 = pf.x + ux * nr, y1 = pf.y + uy * nr;
          const x2 = pt.x - ux * (nr + 7), y2 = pt.y - uy * (nr + 7);
          const strong = dominance > 0.65;
          const sw = Math.min(4, Math.max(1, total * 0.6));
          return (
            <line key={`${from}_${to}`} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={strong ? "var(--accent)" : "var(--neutral)"}
              strokeWidth={sw} opacity={strong ? 0.7 : 0.4}
              markerEnd={strong ? "url(#arr-w)" : "url(#arr-e)"} />
          );
        })}
        {ids.map((id, i) => {
          const { x, y } = pos[id];
          const name = (state.players[id] as { display_name: string })?.display_name ?? id;
          const eloVal = elo[id] ?? 1000;
          const color = PLAYER_COLORS[i % PLAYER_COLORS.length];
          const isSource = sources.includes(id);
          const isSink   = sinks.includes(id);
          return (
            <g key={id}>
              {isSource && <circle cx={x} cy={y} r={nr + 5} fill="none" stroke="var(--win)"  strokeWidth={2} strokeDasharray="4 2" opacity={0.7} />}
              {isSink   && <circle cx={x} cy={y} r={nr + 5} fill="none" stroke="var(--lose)" strokeWidth={2} strokeDasharray="4 2" opacity={0.7} />}
              <circle cx={x} cy={y} r={nr} fill="var(--surface)" stroke={color} strokeWidth={1.5} />
              <text x={x} y={y - 4} textAnchor="middle" fill="var(--text-bright)" fontSize={9} fontFamily="Share Tech Mono" fontWeight={600}>
                {name.slice(0, 7)}
              </text>
              <text x={x} y={y + 8} textAnchor="middle" fill={color} fontSize={8} fontFamily="Share Tech Mono">
                {eloVal}
              </text>
            </g>
          );
        })}
      </svg>
      {(sources.length > 0 || sinks.length > 0) && (
        <div style={{ display: "flex", gap: 20, marginTop: 10, justifyContent: "center", flexWrap: "wrap" }}>
          {sources.map(id => (
            <div key={id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <svg width={12} height={12}><circle cx={6} cy={6} r={5} fill="none" stroke="var(--win)" strokeWidth={1.5} strokeDasharray="3 1.5" /></svg>
              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--win)" }}>
                SOURCE — {(state.players[id] as { display_name: string })?.display_name} beats all opponents in graph
              </span>
            </div>
          ))}
          {sinks.map(id => (
            <div key={id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <svg width={12} height={12}><circle cx={6} cy={6} r={5} fill="none" stroke="var(--lose)" strokeWidth={1.5} strokeDasharray="3 1.5" /></svg>
              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--lose)" }}>
                SINK — {(state.players[id] as { display_name: string })?.display_name} beaten by all opponents in graph
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RollingFormChart({ playerId, games }: { playerId: string; games: Game[] }) {
  const WINDOW = 5;
  const pg = [...games]
    .filter(g => g.winner_id === playerId || g.loser_id === playerId)
    .sort((a, b) => a.timestamp - b.timestamp);
  if (pg.length <= WINDOW) return null;
  const points = pg.slice(WINDOW - 1).map((_, i) => {
    const slice = pg.slice(i, i + WINDOW);
    return slice.filter(x => x.winner_id === playerId).length / WINDOW;
  });
  const W = 500, H = 72, PL = 10, PR = 44, PT = 8, PB = 8;
  const cw = W - PL - PR, ch = H - PT - PB;
  const n = points.length;
  const xS = (i: number) => PL + (n <= 1 ? cw / 2 : (i / (n - 1)) * cw);
  const yS = (v: number) => PT + ch - v * ch;
  const d = points.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i).toFixed(1)},${yS(v).toFixed(1)}`).join(" ");
  const last = points[points.length - 1];
  const lastColor = last > 0.5 ? "var(--win)" : last < 0.5 ? "var(--lose)" : "var(--neutral)";
  return (
    <div className="panel" style={{ padding: "10px 14px", marginBottom: 12 }}>
      <div className="section-label" style={{ marginBottom: 6 }}>ROLLING WIN RATE — last {WINDOW} games</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", overflow: "visible" }}>
        <line x1={PL} y1={yS(0.5)} x2={W - PR} y2={yS(0.5)} stroke="var(--border)" strokeWidth={0.6} strokeDasharray="3 2" />
        <text x={W - PR + 4} y={yS(1) + 3} fill="var(--text-dim)" fontSize={7} fontFamily="Share Tech Mono">100%</text>
        <text x={W - PR + 4} y={yS(0.5) + 3} fill="var(--text-dim)" fontSize={7} fontFamily="Share Tech Mono">50%</text>
        <text x={W - PR + 4} y={yS(0) + 3} fill="var(--text-dim)" fontSize={7} fontFamily="Share Tech Mono">0%</text>
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeLinejoin="round" opacity={0.8} />
        {points.map((v, i) => (
          <circle key={i} cx={xS(i)} cy={yS(v)} r={2.5}
            fill={v > 0.5 ? "var(--win)" : v < 0.5 ? "var(--lose)" : "var(--neutral)"} />
        ))}
        <text x={xS(n - 1) + 5} y={yS(last) + 3} fill={lastColor} fontSize={8} fontFamily="Share Tech Mono">
          {Math.round(last * 100)}%
        </text>
      </svg>
    </div>
  );
}

function RivalryTimeline({ rv, games, players }: {
  rv: { a: string; b: string; aWins: number; bWins: number; total: number };
  games: Game[];
  players: Record<string, { display_name: string }>;
}) {
  const h2hGames = games
    .filter(g =>
      (g.winner_id === rv.a && g.loser_id === rv.b) ||
      (g.winner_id === rv.b && g.loser_id === rv.a)
    )
    .sort((a, b) => a.timestamp - b.timestamp);
  const aName = players[rv.a]?.display_name ?? rv.a;
  const bName = players[rv.b]?.display_name ?? rv.b;
  let aScore = 0, bScore = 0;
  const timeline = h2hGames.map(g => {
    const aWon = g.winner_id === rv.a;
    if (aWon) aScore++; else bScore++;
    return { g, aWon, aScore, bScore };
  });
  return (
    <div className="panel fade-in" style={{ padding: "12px 16px", borderTop: "none", marginTop: -1, marginBottom: 4 }}>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 12 }}>
        {timeline.map(({ g, aWon, aScore, bScore }) => (
          <div key={g.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
            <div style={{
              width: 32, height: 32, borderRadius: "50%", opacity: 0.9,
              background: aWon ? "var(--win)" : "var(--lose)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "0.67rem", fontFamily: "Share Tech Mono", color: "#000", fontWeight: 700,
            }}>
              {(aWon ? aName : bName).slice(0, 3).toUpperCase()}
            </div>
            <span className="font-mono" style={{ fontSize: "0.63rem", color: "var(--text-dim)" }}>{aScore}–{bScore}</span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {timeline.map(({ g, aWon }, idx) => {
          const ws = g.winner_stats, ls = g.loser_stats;
          const wKD = ws.deaths > 0 ? (ws.kills / ws.deaths).toFixed(2) : ws.kills.toFixed(0);
          const lKD = ls.deaths > 0 ? (ls.kills / ls.deaths).toFixed(2) : ls.kills.toFixed(0);
          const winName = aWon ? aName : bName;
          const loseName = aWon ? bName : aName;
          return (
            <div key={g.id} className="panel" style={{ padding: "6px 10px", display: "flex", alignItems: "center", gap: 8 }}>
              <span className="font-mono" style={{ fontSize: "0.67rem", color: "var(--text-dim)", minWidth: 18 }}>#{idx + 1}</span>
              <span className="font-mono" style={{ fontSize: "0.75rem", color: aWon ? "var(--win)" : "var(--lose)", flex: 1 }}>{winName}</span>
              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>{ws.kills}K/{ws.deaths}D · KD {wKD}</span>
              <span className="font-mono" style={{ fontSize: "0.67rem", color: "var(--accent)" }}>DEF</span>
              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>{ls.kills}K/{ls.deaths}D · KD {lKD}</span>
              <span className="font-mono" style={{ fontSize: "0.75rem", color: aWon ? "var(--lose)" : "var(--win)", flex: 1, textAlign: "right" }}>{loseName}</span>
              <span className="font-mono" style={{ fontSize: "0.65rem", color: "var(--text-dim)", minWidth: 56, textAlign: "right" }}>
                {new Date(g.timestamp).toLocaleDateString()}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function computeRecentStats(playerId: string, games: Game[], window = 5) {
  const pg = [...games]
    .filter(g => g.winner_id === playerId || g.loser_id === playerId)
    .sort((a, b) => a.timestamp - b.timestamp);
  if (pg.length < window + 1) return null;
  function agg(gs: typeof pg) {
    let kills = 0, deaths = 0, wins = 0;
    for (const g of gs) {
      const won = g.winner_id === playerId;
      const s = won ? g.winner_stats : g.loser_stats;
      kills += s.kills; deaths += s.deaths;
      if (won) wins++;
    }
    return {
      kd: deaths > 0 ? kills / deaths : kills,
      win_rate: wins / gs.length,
    };
  }
  return { recent: agg(pg.slice(-window)), overall: agg(pg), games: pg.length };
}

function computeStreak(playerId: string, games: Game[]) {
  const pg = [...games]
    .filter(g => g.winner_id === playerId || g.loser_id === playerId)
    .sort((a, b) => b.timestamp - a.timestamp);
  if (pg.length === 0) return { count: 0, win: true, form: [] as boolean[] };
  const form = pg.slice(0, 5).map(g => g.winner_id === playerId);
  const win = form[0];
  let count = 0;
  for (const g of pg) { if ((g.winner_id === playerId) === win) count++; else break; }
  return { count, win, form };
}

function computeArchetype(playerId: string, ranking: RankEntry[], state: GraphState, eloHistory?: Record<string, Array<{ timestamp: number; elo: number }>>): string {
  const rank = ranking.findIndex(r => r.player_id === playerId);
  const n = ranking.length;
  const entry = ranking[rank];
  if (!entry) return "";
  const sv = entry.stat_vec;
  if (sv.games_played < 5) return "THE UNKNOWN";

  const games = [...state.games]
    .sort((a, b) => a.timestamp - b.timestamp)
    .filter(g => g.winner_id === playerId || g.loser_id === playerId);

  const topIds = ranking.slice(0, Math.ceil(n / 2)).map(r => r.player_id).filter(id => id !== playerId);
  const botIds  = ranking.slice(Math.ceil(n / 2)).map(r => r.player_id).filter(id => id !== playerId);

  const vsTop = games.filter(g => topIds.includes(g.winner_id === playerId ? g.loser_id : g.winner_id));
  const vsBot = games.filter(g => botIds.includes(g.winner_id === playerId ? g.loser_id : g.winner_id));
  const wrTop = vsTop.length > 0 ? vsTop.filter(g => g.winner_id === playerId).length / vsTop.length : -1;
  const wrBot = vsBot.length > 0 ? vsBot.filter(g => g.winner_id === playerId).length / vsBot.length : -1;

  const outcomes = games.map(g => g.winner_id === playerId);
  const alts = outcomes.slice(1).filter((v, i) => v !== outcomes[i]).length;
  const streakRatio = outcomes.length > 1 ? alts / (outcomes.length - 1) : 1;

  const maxGames = Math.max(...ranking.map(r => r.stat_vec.games_played));

  // ELO velocity slope
  let eloSlope = 0;
  if (eloHistory) {
    const h = eloHistory[playerId] ?? [];
    const recent = h.slice(-5);
    if (recent.length >= 2) {
      const xMean = (recent.length - 1) / 2;
      const yMean = recent.reduce((s, p) => s + p.elo, 0) / recent.length;
      const num = recent.reduce((s, p, i) => s + (i - xMean) * (p.elo - yMean), 0);
      const den = recent.reduce((s, _, i) => s + (i - xMean) ** 2, 0);
      eloSlope = den > 0 ? num / den : 0;
    }
  }

  if (rank === 0 && sv.win_rate > 0.6) return "THE TYRANT";
  if (eloSlope >= 3.0) return "THE CLIMBER";
  if (sv.kd >= 3.0 && sv.win_rate < 0.5) return "THE FRAGGER";
  if (sv.kd <= 1.5 && sv.win_rate > 0.55) return "THE TACTICIAN";
  if (vsTop.length >= 2 && vsBot.length >= 2 && wrTop >= 0 && wrBot >= 0 && wrTop - wrBot > 0.3) return "GIANT KILLER";
  if (vsTop.length >= 2 && vsBot.length >= 2 && wrTop >= 0 && wrBot >= 0 && wrBot - wrTop > 0.3) return "THE BULLY";
  if (games.length >= 5 && streakRatio < 0.3) return "THE STREAKY";
  if (sv.games_played >= maxGames && sv.games_played >= 4) return "THE GRINDER";
  if (sv.kd < 1.2 && sv.win_rate > 0.55) return "THE CLOSER";
  return "THE SOLDIER";
}

function predictScore(
  state: GraphState,
  aId: string,
  bId: string,
  p_a_wins: number,
): { aKills: number; bKills: number } {
  const games = state.games;
  if (games.length === 0) return { aKills: 7, bKills: 5 };

  const globalWinKills  = games.reduce((s, g) => s + g.winner_stats.kills, 0) / games.length;
  const globalLoseKills = games.reduce((s, g) => s + g.loser_stats.kills,  0) / games.length;

  const stats = (id: string) => {
    const won  = games.filter(g => g.winner_id === id);
    const lost = games.filter(g => g.loser_id  === id);
    return {
      win:  won.length  >= 2 ? won.reduce( (s, g) => s + g.winner_stats.kills, 0) / won.length  : null,
      lose: lost.length >= 2 ? lost.reduce((s, g) => s + g.loser_stats.kills,  0) / lost.length : null,
    };
  };

  const a = stats(aId), b = stats(bId);
  const blend = (personal: number | null, global: number) =>
    personal !== null ? personal * 0.65 + global * 0.35 : global;

  let aKills: number, bKills: number;
  if (p_a_wins >= 0.5) {
    aKills = Math.round(blend(a.win,  globalWinKills));
    bKills = Math.round(blend(b.lose, globalLoseKills));
    if (bKills >= aKills) bKills = aKills - 1;
  } else {
    bKills = Math.round(blend(b.win,  globalWinKills));
    aKills = Math.round(blend(a.lose, globalLoseKills));
    if (aKills >= bKills) aKills = bKills - 1;
  }

  return { aKills: Math.max(0, aKills), bKills: Math.max(0, bKills) };
}

function generatePowerBlurb(
  ranking: RankEntry[],
  state: GraphState,
  eloHistory: Record<string, Array<{ timestamp: number; elo: number }>>,
): string {
  if (ranking.length < 2 || state.games.length < 3) return "";

  const top = ranking[0];

  let bigMoverName = "";
  let bigMoverGain = 0;
  for (const r of ranking) {
    const h = eloHistory[r.player_id] ?? [];
    if (h.length >= 2) {
      const gain = h[h.length - 1].elo - h[Math.max(0, h.length - 5)].elo;
      if (gain > bigMoverGain) { bigMoverGain = gain; bigMoverName = r.display_name; }
    }
  }

  const streaks = ranking.map(r => {
    const pg = [...state.games]
      .filter(g => g.winner_id === r.player_id || g.loser_id === r.player_id)
      .sort((a, b) => b.timestamp - a.timestamp);
    let wins = 0, losses = 0;
    for (const g of pg) {
      if (g.winner_id === r.player_id) { if (losses) break; wins++; }
      else { if (wins) break; losses++; }
    }
    return { name: r.display_name, wins, losses };
  });

  const hottest = streaks.filter(s => s.wins  >= 2).sort((a, b) => b.wins   - a.wins)[0];
  const coldest = streaks.filter(s => s.losses >= 2).sort((a, b) => b.losses - a.losses)[0];

  const parts: string[] = [];

  const topWR = Math.round(top.stat_vec.win_rate * 100);
  const topStreakEntry = streaks.find(s => s.name === top.display_name);
  const topStreakNote = topStreakEntry && topStreakEntry.wins >= 3 ? `, riding a ${topStreakEntry.wins}-game win streak` : "";
  parts.push(`${top.display_name} sits at #1 with a ${topWR}% win rate${topStreakNote}.`);

  if (bigMoverName && bigMoverName !== top.display_name && bigMoverGain >= 12) {
    parts.push(`${bigMoverName} is the biggest mover, up +${Math.round(bigMoverGain)} ELO recently.`);
  }

  if (hottest && hottest.name !== top.display_name && hottest.wins >= 3) {
    parts.push(`${hottest.name} is running hot — ${hottest.wins} straight wins.`);
  } else if (coldest && coldest.losses >= 3) {
    parts.push(`${coldest.name} is on a ${coldest.losses}-game skid${coldest.losses >= 5 ? " — someone call a medic" : ""}.`);
  }

  return parts.join(" ");
}

function EloSparkline({ history }: { history: Array<{ timestamp: number; elo: number }> }) {
  if (history.length < 2) return null;
  const W = 48, H = 14;
  const vals = history.map(p => p.elo);
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 30;
  const xS = (i: number) => (i / (vals.length - 1)) * W;
  const yS = (v: number) => H - ((v - min) / range) * (H - 2) - 1;
  const d = vals.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i).toFixed(1)},${yS(v).toFixed(1)}`).join(" ");
  const last = vals[vals.length - 1], first = vals[0];
  const color = last > first + 5 ? "var(--win)" : last < first - 5 ? "var(--lose)" : "var(--text-dim)";
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block", marginLeft: "auto", overflow: "visible" }}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.2} strokeLinejoin="round" opacity={0.75} />
      <circle cx={xS(vals.length - 1)} cy={yS(last)} r={1.8} fill={color} />
    </svg>
  );
}

function wilsonCI(wins: number, total: number, z = 1.96): [number, number] {
  if (total === 0) return [0, 1];
  const p = wins / total;
  const z2 = z * z;
  const center = (p + z2 / (2 * total)) / (1 + z2 / total);
  const hw = z * Math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total);
  return [Math.max(0, center - hw), Math.min(1, center + hw)];
}

// Epistemic CI: width reflects actual evidence quality, not MC sampling error.
// Direct games → tight; low Katz mass (purely indirect) → wide.
function katzCI(p: number, directGames: number, evidenceMass: number): [number, number] {
  let hw: number;
  if (directGames >= 6) hw = 0.04;
  else if (directGames >= 3) hw = 0.05 + (6 - Math.min(6, directGames)) * 0.01;
  else if (directGames === 2) hw = 0.09;
  else if (directGames === 1) hw = 0.13;
  else hw = Math.max(0.14, 0.28 * Math.exp(-evidenceMass * 1.5));
  return [Math.max(0, p - hw), Math.min(1, p + hw)];
}

function PlacementSparkline({ dist }: { dist: number[] }) {
  const n = dist.length;
  if (n === 0 || dist.every(v => v === 0)) return null;
  const max = Math.max(...dist);
  const W = 60, H = 16;
  const bw = W / n;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block", marginLeft: "auto", marginTop: 3, overflow: "visible" }}>
      {dist.map((count, rank) => {
        const h = max > 0 ? Math.max(1, (count / max) * (H - 2)) : 0;
        const x = rank * bw;
        const frac = n > 1 ? rank / (n - 1) : 0;
        const color = frac < 0.4 ? "var(--win)" : frac < 0.7 ? "var(--neutral)" : "var(--lose)";
        const opacity = 0.4 + (1 - frac) * 0.55;
        return (
          <rect key={rank} x={x + 0.5} y={H - h} width={Math.max(1, bw - 1.5)} height={h}
            fill={color} opacity={opacity} />
        );
      })}
    </svg>
  );
}

function PlayerScatterPlot({ ranking }: { ranking: RankEntry[] }) {
  const pts = ranking.filter(r => r.stat_vec.games_played >= 1);
  if (pts.length < 2) return null;
  const W = 560, H = 260, PL = 44, PR = 16, PT = 20, PB = 36;
  const cw = W - PL - PR, ch = H - PT - PB;

  const kds = pts.map(r => r.stat_vec.kd);
  const wrs = pts.map(r => r.stat_vec.win_rate);
  const gps = pts.map(r => r.stat_vec.games_played);
  const kdPad = Math.max(0.4, (Math.max(...kds) - Math.min(...kds)) * 0.18);
  const wrPad = 0.1;
  const xMin = Math.min(...kds) - kdPad, xMax = Math.max(...kds) + kdPad;
  const yMin = Math.max(0, Math.min(...wrs) - wrPad), yMax = Math.min(1, Math.max(...wrs) + wrPad);
  const xS = (v: number) => PL + ((v - xMin) / (xMax - xMin)) * cw;
  const yS = (v: number) => PT + ch - ((v - yMin) / (yMax - yMin)) * ch;
  const maxGP = Math.max(...gps);
  const bR = (gp: number) => Math.max(9, Math.min(22, 9 + (gp / maxGP) * 13));

  const kdTicks = [Math.ceil(xMin * 2) / 2, 1, Math.floor(xMax * 2) / 2].filter((v, i, a) => a.indexOf(v) === i && v >= xMin && v <= xMax);
  const wrTicks = [0.25, 0.5, 0.75].filter(v => v >= yMin && v <= yMax);

  return (
    <div>
      <div className="section-label" style={{ marginBottom: 8 }}>PLAYER LANDSCAPE — KD × WIN RATE (bubble = games played)</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", overflow: "visible" }}>
        {/* Grid */}
        {wrTicks.map(v => (
          <g key={v}>
            <line x1={PL} y1={yS(v)} x2={PL + cw} y2={yS(v)} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 2" />
            <text x={PL - 4} y={yS(v)} fill="var(--text-dim)" fontSize={7} fontFamily="Share Tech Mono" textAnchor="end" dominantBaseline="middle">{Math.round(v * 100)}%</text>
          </g>
        ))}
        {kdTicks.map(v => (
          <g key={v}>
            <line x1={xS(v)} y1={PT} x2={xS(v)} y2={PT + ch} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 2" />
            <text x={xS(v)} y={PT + ch + 10} fill="var(--text-dim)" fontSize={7} fontFamily="Share Tech Mono" textAnchor="middle">{v.toFixed(1)}</text>
          </g>
        ))}
        {/* Quadrant labels */}
        {xMin < 1 && <text x={xS(xMin + 0.01)} y={PT + 8} fill="var(--lose)" fontSize={7} fontFamily="Share Tech Mono" opacity={0.45}>STRUGGLING</text>}
        {xMax > 1 && <text x={xS(xMax - 0.01)} y={PT + 8} fill="var(--win)" fontSize={7} fontFamily="Share Tech Mono" textAnchor="end" opacity={0.45}>DOMINANT</text>}
        {/* Axis labels */}
        <text x={PL + cw / 2} y={H - 4} fill="var(--text-dim)" fontSize={8} fontFamily="Share Tech Mono" textAnchor="middle">K/D RATIO</text>
        <text x={10} y={PT + ch / 2} fill="var(--text-dim)" fontSize={8} fontFamily="Share Tech Mono" textAnchor="middle" transform={`rotate(-90, 10, ${PT + ch / 2})`}>WIN RATE</text>
        {/* Bubbles */}
        {pts.map((r, i) => {
          const x = xS(r.stat_vec.kd), y = yS(r.stat_vec.win_rate), radius = bR(r.stat_vec.games_played);
          const color = PLAYER_COLORS[i % PLAYER_COLORS.length];
          const label = r.display_name.length > 7 ? r.display_name.slice(0, 6) + "…" : r.display_name;
          return (
            <g key={r.player_id}>
              <circle cx={x} cy={y} r={radius} fill={color} opacity={0.12} />
              <circle cx={x} cy={y} r={radius} fill="none" stroke={color} strokeWidth={1.5} opacity={0.8} />
              <text x={x} y={y + 3} textAnchor="middle" fill={color} fontSize={7.5} fontFamily="Share Tech Mono" fontWeight={600}>{label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function StatInput({ label, stats, onChange }: {
  label: string; stats: PerGameStats; onChange: (s: PerGameStats) => void;
}) {
  const fields: (keyof PerGameStats)[] = ["kills", "deaths"];
  return (
    <div style={{ flex: 1 }}>
      <div className="section-label" style={{ marginBottom: 8 }}>{label}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {fields.map(f => (
          <div key={f}>
            <div style={{ fontSize: "0.70rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", letterSpacing: "0.1em", marginBottom: 2 }}>{f.toUpperCase()}</div>
            <input className="input" type="number" min="0" max="99" value={stats[f] || ""}
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
  const [error, setError] = useState("");
  const [elo, setElo] = useState<Record<string, number>>({});
  const [predAccuracy, setPredAccuracy] = useState<{ correct: number; total: number } | null>(null);
  const [eloHistory, setEloHistory] = useState<Record<string, Array<{ timestamp: number; elo: number }>>>({});
  const [rankSort, setRankSort] = useState<"tour" | "elo">("tour");
  const [metaStability, setMetaStability] = useState<number | null>(null);
  const [selectedRivalry, setSelectedRivalry] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [h2hSim, setH2hSim] = useState<H2hSim>({});
  const [champCounts, setChampCounts] = useState<Record<string, number>>({});
  const [btResult, setBtResult] = useState<BTResult | null>(null);
  const [cycleAnalysis, setCycleAnalysis] = useState<CycleAnalysis | null>(null);
  const [looCv, setLooCv] = useState<LooCvResult | null>(null);
  const [skillGap, setSkillGap] = useState<SkillGapTrend | null>(null);
  const [formRating, setFormRating] = useState<Record<string, number>>({});
  const [suggestSort, setSuggestSort] = useState<"closest" | "elovalue" | "uncertain">("closest");
  const [scoreWinner, setScoreWinner] = useState<string>("");
  const [scoreLoser, setScoreLoser] = useState<string>("");

  useEffect(() => {
    const saved = localStorage.getItem("graphelo-theme") as "dark" | "light" | null;
    if (saved) setTheme(saved);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "light") {
      document.body.classList.add("light");
      for (const [k, v] of Object.entries(LIGHT_VARS)) root.style.setProperty(k, v);
    } else {
      document.body.classList.remove("light");
      for (const k of Object.keys(LIGHT_VARS)) root.style.removeProperty(k);
    }
  }, [theme]);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("graphelo-theme", next);
  }

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

  // Players tab
  const [newName, setNewName] = useState("");
  const [addingPlayer, setAddingPlayer] = useState(false);

  // History tab
  const [historyFilter, setHistoryFilter] = useState("");
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);

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

  const computeRanking = useCallback((s: GraphState) => {
    try {
      const result = simulateRoundRobin(s);
      setRanking(result.ranking);
      setH2hSim(result.h2h_sim);
      setChampCounts(result.champ_counts);
      const eloMap = computeElo(s);
      setElo(eloMap);
      const hist = computeEloHistory(s);
      setEloHistory(hist);
      setMetaStability(computeMetaStability(s));
      if (s.games.length > 1) setPredAccuracy(computePredictionAccuracy(s));
      setBtResult(computeBradleyTerry(s));
      setCycleAnalysis(computeCycles(s));
      setLooCv(computeLooCvBrier(s));
      setSkillGap(computeSkillGapTrend(s));
      setFormRating(computeFormRating(s));
    } catch { setError("Ranking failed"); }
  }, []);

  useEffect(() => {
    refresh().then(s => { if (s) computeRanking(s); });
  }, [refresh, computeRanking]);

  const players = Object.values(state.players);

  const sortedRanking = useMemo(() =>
    rankSort === "elo"
      ? [...ranking].sort((a, b) => (elo[b.player_id] ?? 1000) - (elo[a.player_id] ?? 1000))
      : ranking,
    [ranking, rankSort, elo],
  );

  const rivalries = useMemo(() => {
    const pairs: Record<string, { a: string; b: string; aWins: number; bWins: number; total: number }> = {};
    for (const g of state.games) {
      const [a, b] = [g.winner_id, g.loser_id].sort();
      const key = `${a}_${b}`;
      if (!pairs[key]) pairs[key] = { a, b, aWins: 0, bWins: 0, total: 0 };
      pairs[key].total++;
      if (g.winner_id === a) pairs[key].aWins++; else pairs[key].bWins++;
    }
    return Object.values(pairs)
      .filter(rv => rv.total >= 2 && state.players[rv.a] && state.players[rv.b])
      .sort((x, y) => y.total - x.total || Math.abs(x.aWins - x.bWins) - Math.abs(y.aWins - y.bWins))
      .slice(0, 5);
  }, [state]);

  const upsetMap = useMemo(() => {
    const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
    const map: Record<string, boolean> = {};
    for (let i = 1; i < sorted.length; i++) {
      const g = sorted[i];
      if (!state.players[g.winner_id] || !state.players[g.loser_id]) continue;
      const hist = { players: state.players, games: sorted.slice(0, i) };
      const { p_a_wins } = predictPairwise(hist, g.winner_id, g.loser_id);
      if (p_a_wins < 0.35) map[g.id] = true;
    }
    return map;
  }, [state]);

  const revengeMap = useMemo(() => {
    const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
    const map: Record<string, boolean> = {};
    for (let i = 1; i < sorted.length; i++) {
      const { id, winner_id, loser_id } = sorted[i];
      for (let j = i - 1; j >= 0; j--) {
        const prev = sorted[j];
        const involves = (prev.winner_id === winner_id && prev.loser_id === loser_id) ||
                         (prev.winner_id === loser_id  && prev.loser_id === winner_id);
        if (involves) { if (prev.winner_id === loser_id) map[id] = true; break; }
      }
    }
    return map;
  }, [state]);

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
      else { const s = await res.json(); setState(s); computeRanking(s); setNewName(""); }
    } finally { setAddingPlayer(false); }
  }

  async function handleDeletePlayer(id: string) {
    const name = state.players[id]?.display_name ?? id;
    const gameCount = state.games.filter(g => g.winner_id === id || g.loser_id === id).length;
    if (!confirm(`Delete ${name}? This removes them and all ${gameCount} of their game(s). Cannot be undone.`)) return;
    const res = await fetch(`/api/players/${id}`, { method: "DELETE" });
    if (!res.ok) { const d = await res.json(); setError(d.error); return; }
    const s = await res.json();
    setState(s);
    computeRanking(s);
    if (historyFilter === id) setHistoryFilter("");
    if (predA === id) setPredA("");
    if (predB === id) setPredB("");
  }

  async function handleDeleteGame(id: string) {
    if (!confirm("Delete this game? This cannot be undone.")) return;
    const res = await fetch(`/api/matches/${id}`, { method: "DELETE" });
    if (!res.ok) { const d = await res.json(); setError(d.error); return; }
    const s = await res.json();
    setState(s);
    computeRanking(s);
  }

  async function logGame(e: React.FormEvent) {
    e.preventDefault();
    if (!logWinner || !logLoser) return;
    setLogging(true); setLogMsg("");
    try {
      const sw = parseInt(scoreWinner), sl = parseInt(scoreLoser);
      const body: Record<string, unknown> = { winner_id: logWinner, loser_id: logLoser, winner_stats: winStats, loser_stats: loseStats };
      if (!isNaN(sw) && !isNaN(sl) && sw > sl) { body.score_winner = sw; body.score_loser = sl; }
      const res = await fetch("/api/matches", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) { const d = await res.json(); setError(d.error); }
      else {
        const s = await res.json();
        setLogMsg(`✓ ${state.players[logWinner]?.display_name} def. ${state.players[logLoser]?.display_name}`);
        setLogWinner(""); setLogLoser("");
        setWinStats(EMPTY_STATS()); setLoseStats(EMPTY_STATS());
        setScoreWinner(""); setScoreLoser("");
        setState(s);
        computeRanking(s);
      }
    } finally { setLogging(false); }
  }

  function getPrediction() {
    if (!predA || !predB || predA === predB) return;
    setPrediction(predictPairwise(state, predA, predB));
  }

  function pct(n: number, dec = 0) {
    return `${(n * 100).toFixed(dec)}%`;
  }
  function fmt(n: number, dec = 2) { return n.toFixed(dec); }
  function confLabel(c: number) { return c < 0.25 ? "LOW" : c < 0.6 ? "MED" : "HIGH"; }
  function confColor(c: number) { return c < 0.25 ? "var(--lose)" : c < 0.6 ? "var(--neutral)" : "var(--win)"; }

  const mostInteresting = (() => {
    if (ranking.length < 2) return null;
    let best: { a: RankEntry; b: RankEntry; p: number } | null = null;
    let minDiff = 1;
    for (let i = 0; i < ranking.length; i++) {
      for (let j = i + 1; j < ranking.length; j++) {
        const a = ranking[i], b = ranking[j];
        const p = a.matchup_table[b.player_id] ?? 0.5;
        const diff = Math.abs(p - 0.5);
        if (diff < minDiff) { minDiff = diff; best = { a, b, p }; }
      }
    }
    return best;
  })();

  const allGamesReversed = [...state.games].reverse();
  const filteredGames = historyFilter
    ? allGamesReversed.filter(g => g.winner_id === historyFilter || g.loser_id === historyFilter)
    : allGamesReversed.slice(0, 30);
  const winsGames   = historyFilter ? allGamesReversed.filter(g => g.winner_id === historyFilter) : [];
  const lossesGames = historyFilter ? allGamesReversed.filter(g => g.loser_id  === historyFilter) : [];

  return (
    <div style={{ minHeight: "100vh", padding: "24px", maxWidth: 900, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <div className="live-indicator" />
          <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--accent)", letterSpacing: "0.2em" }}>
            GRAPHELO // R6 CUSTOM 1v1
          </span>
          <button onClick={toggleTheme} className="btn" style={{ marginLeft: "auto", padding: "1px 8px", fontSize: "0.72rem", letterSpacing: "0.12em" }}>
            {theme === "dark" ? "◑ LIGHT" : "◑ DARK"}
          </button>
        </div>
        <div className="gradient-bar" style={{ marginBottom: 8 }} />
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1 className="font-display" style={{ fontSize: "2rem", fontWeight: 700, color: "var(--text-bright)", letterSpacing: "0.05em" }}>
            GRAPH<span style={{ color: "var(--accent)" }}>ELO</span>
          </h1>
          <span className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>
            {players.length} PLAYERS · {state.games.length} GAMES
          </span>
        </div>
      </div>

      {error && (
        <div style={{ marginBottom: 14, padding: "7px 12px", border: "1px solid var(--lose)", color: "var(--lose)", fontSize: "0.86rem", fontFamily: "Share Tech Mono", display: "flex", justifyContent: "space-between" }}>
          <span>⚠ {error}</span>
          <button onClick={() => setError("")} style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer" }}>×</button>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--border)", marginBottom: 20 }}>
        {(["ranking", "log", "matchup", "history", "players", "system"] as Tab[]).map(t => (
          <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {loading && <div className="font-mono" style={{ color: "var(--text-dim)", fontSize: "0.8rem" }}>LOADING...</div>}

      {/* ── RANKING ── */}
      {!loading && tab === "ranking" && (
        <div className="fade-in">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8, flexWrap: "wrap" }}>
            <div className="section-label">ROUND ROBIN RANKING</div>
            <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>KATZ β=0.5 · 1000-MC</span>
            {looCv && looCv.n >= 3 && (
              <span className="font-mono" style={{ fontSize: "0.72rem", color: looCv.brier_skill > 0.1 ? "var(--win)" : "var(--text-dim)" }}>
                LOO {Math.round(looCv.accuracy * 100)}% · BRIER {looCv.brier_score.toFixed(2)} · SKILL +{Math.round(looCv.brier_skill * 100)}%
              </span>
            )}
            {!looCv && predAccuracy && predAccuracy.total >= 3 && (
              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                IN-SAMPLE {Math.round(predAccuracy.correct / predAccuracy.total * 100)}% ({predAccuracy.correct}/{predAccuracy.total})
              </span>
            )}
            {metaStability !== null && (
              <span className="font-mono" style={{ fontSize: "0.72rem", color: metaStability > 0.7 ? "var(--win)" : metaStability > 0.4 ? "var(--neutral)" : "var(--lose)" }}>
                META {Math.round(metaStability * 100)}% {metaStability > 0.7 ? "STABLE" : metaStability > 0.4 ? "SHIFTING" : "VOLATILE"}
              </span>
            )}
            <div style={{ marginLeft: "auto", display: "flex", gap: 5, alignItems: "center" }}>
              {(["tour", "elo"] as const).map(s => (
                <button key={s} className="btn" onClick={() => setRankSort(s)} style={{ padding: "2px 8px", fontSize: "0.73rem", background: rankSort === s ? "var(--accent)" : undefined, color: rankSort === s ? "#000" : undefined }}>
                  {s === "tour" ? "CHAMP%" : "ELO"}
                </button>
              ))}
              <button className="btn" style={{ padding: "3px 8px", fontSize: "0.7rem" }} onClick={() => computeRanking(state)}>↺</button>
            </div>
          </div>
          {/* Parity + skill gap banner */}
          {Object.keys(state.players).length >= 2 && state.games.length >= 4 && (() => {
            const parity = computeParityIndex(state, champCounts, 1000);
            return (
              <div style={{ display: "flex", gap: 16, marginBottom: 14, flexWrap: "wrap" }}>
                <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                  PARITY <span style={{ color: parity.normalized_parity > 0.7 ? "var(--win)" : parity.normalized_parity > 0.4 ? "var(--neutral)" : "var(--lose)" }}>{Math.round(parity.normalized_parity * 100)}%</span>
                  {" · "}ELO SPREAD <span style={{ color: "var(--text-bright)" }}>±{Math.round(parity.elo_spread)}pts</span>
                  {skillGap && <> {" · "}FIELD <span style={{ color: skillGap.trend === "CONVERGING" ? "var(--win)" : skillGap.trend === "DIVERGING" ? "var(--lose)" : "var(--text-dim)" }}>{skillGap.trend} {skillGap.trend === "CONVERGING" ? "↓" : skillGap.trend === "DIVERGING" ? "↑" : "→"}</span></>}
                </span>
              </div>
            );
          })()}

          {mostInteresting && state.games.length > 0 && (
            <div className="panel" style={{ padding: "11px 16px", marginBottom: 14, display: "flex", alignItems: "center", gap: 14, borderColor: "var(--accent)" }}>
              <div style={{ flex: 1 }}>
                <div className="section-label" style={{ marginBottom: 5, color: "var(--accent)" }}>CLOSEST MATCHUP</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span className="player-name">{mostInteresting.a.display_name}</span>
                  <span className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-dim)" }}>
                    {pct(mostInteresting.p, 1)} · {pct(1 - mostInteresting.p, 1)}
                  </span>
                  <span className="player-name">{mostInteresting.b.display_name}</span>
                </div>
              </div>
              <button className="btn" style={{ padding: "3px 10px", fontSize: "0.7rem", flexShrink: 0 }}
                onClick={() => { setPredA(mostInteresting!.a.player_id); setPredB(mostInteresting!.b.player_id); setPrediction(predictPairwise(state, mostInteresting!.a.player_id, mostInteresting!.b.player_id)); setTab("matchup"); }}>
                ANALYZE →
              </button>
            </div>
          )}

          {ranking.length === 0 || state.games.length === 0 ? (
            <div style={{ color: "var(--text-dim)", fontSize: "0.92rem" }}>
              {players.length < 2 ? "Add at least 2 players and log a game." : "Log some games to generate a ranking."}
            </div>
          ) : (
            <>
              {/* Power rankings blurb */}
              {(() => {
                const blurb = generatePowerBlurb(ranking, state, eloHistory);
                return blurb ? (
                  <div className="panel" style={{ padding: "10px 14px", marginBottom: 10, borderColor: "var(--border2)" }}>
                    <div className="section-label" style={{ marginBottom: 5, color: "var(--accent)" }}>POWER RANKINGS</div>
                    <p className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text)", lineHeight: 1.65 }}>{blurb}</p>
                  </div>
                ) : null;
              })()}

              {/* Table header */}
              <div style={{ display: "grid", gridTemplateColumns: "28px 1fr 64px 72px 56px 64px 64px 72px", gap: 10, padding: "5px 12px", marginBottom: 4 }}>
                {["#", "PLAYER", "WIN%", "CHAMP%", "ELO", "KD", "KPG", "W-L"].map((h, i) => (
                  <span key={h} className="section-label" style={{ textAlign: i > 1 ? "right" : "left" }}>{h}</span>
                ))}
              </div>

              {sortedRanking.map((r, i) => {
                const sv = r.stat_vec;
                const isTop = i === 0;
                const streak = computeStreak(r.player_id, state.games);
                const isSelected = selectedPlayer === r.player_id;
                const archetype = computeArchetype(r.player_id, sortedRanking, state, eloHistory);
                return (
                  <div key={r.player_id} style={{ marginBottom: 1 }}>
                    <div className="panel corner-tl" onClick={() => setSelectedPlayer(isSelected ? null : r.player_id)} style={{
                      display: "grid",
                      gridTemplateColumns: "28px 1fr 64px 72px 56px 64px 64px 72px",
                      gap: 10, padding: "13px 14px", alignItems: "center",
                      cursor: "pointer",
                      borderColor: isSelected ? "var(--accent)" : isTop ? "var(--accent)" : undefined,
                    }}>
                      <span className="rank-number" style={{ color: isTop ? "var(--accent)" : undefined, fontWeight: isTop ? 700 : undefined }}>
                        {i + 1}
                      </span>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div className="player-name" style={{ color: isTop ? "var(--accent)" : undefined }}>{r.display_name}</div>
                          {streak.count > 0 && (
                            <span className="font-mono" style={{ fontSize: "0.67rem", padding: "1px 4px", border: "1px solid", borderColor: streak.win ? "var(--win)" : "var(--lose)", color: streak.win ? "var(--win)" : "var(--lose)", lineHeight: 1.4, flexShrink: 0 }}>
                              {streak.win ? "W" : "L"}{streak.count}
                            </span>
                          )}
                        </div>
                        {archetype && (
                          <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--accent2)", letterSpacing: "0.1em", marginTop: 3 }}>
                            {archetype}
                          </div>
                        )}
                        <div style={{ display: "flex", gap: 3, marginTop: 4 }}>
                          {streak.form.map((w, idx) => (
                            <div key={idx} style={{ width: 6, height: 6, borderRadius: "50%", background: w ? "var(--win)" : "var(--lose)" }} />
                          ))}
                        </div>
                      </div>
                      <div style={{ textAlign: "right" }} className="winrate">
                        {sv.games_played > 0 ? pct(sv.win_rate, 0) : "—"}
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <span className="rating-value" style={{ fontSize: "0.95rem" }}>{pct(r.tournament_win_pct, 1)}</span>
                        <PlacementSparkline dist={r.placement_dist} />
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 4 }}>
                          {(() => {
                            const rel = computeReliability(r.player_id, state.games);
                            const dotColor = rel < 0.3 ? "var(--lose)" : rel < 0.65 ? "var(--neutral)" : "var(--win)";
                            return <div style={{ width: 5, height: 5, borderRadius: "50%", background: dotColor, flexShrink: 0 }} title={`Reliability: ${Math.round(rel * 100)}%`} />;
                          })()}
                          <div className="winrate">{elo[r.player_id] ?? 1000}</div>
                          {(() => {
                            const vel = computeEloVelocity(eloHistory[r.player_id] ?? []);
                            if (!vel) return null;
                            const arrow = vel.slope > 5 ? "↑" : vel.slope < -5 ? "↓" : "→";
                            const color = vel.slope > 5 ? "var(--win)" : vel.slope < -5 ? "var(--lose)" : "var(--text-dim)";
                            return <span className="font-mono" style={{ fontSize: "0.75rem", color, lineHeight: 1 }}>{arrow}</span>;
                          })()}
                        </div>
                        <EloSparkline history={eloHistory[r.player_id] ?? []} />
                      </div>
                      <div style={{ textAlign: "right" }} className="winrate">{fmt(sv.kd)}</div>
                      <div style={{ textAlign: "right" }} className="winrate">{fmt(sv.kpr, 1)}</div>
                      {(() => {
                        const w = state.games.filter(g => g.winner_id === r.player_id).length;
                        const l = state.games.filter(g => g.loser_id  === r.player_id).length;
                        return (
                          <div style={{ textAlign: "right" }} className="winrate">
                            <span style={{ color: "var(--win)" }}>{w}</span>
                            <span style={{ color: "var(--text-dim)" }}>–</span>
                            <span style={{ color: "var(--lose)" }}>{l}</span>
                          </div>
                        );
                      })()}
                    </div>
                    {isSelected && (() => {
                      const opponents = players.filter(p => p.id !== r.player_id);
                      const rows = opponents.map(opp => {
                        const pred = predictPairwise(state, r.player_id, opp.id);
                        const wins = state.games.filter(g => g.winner_id === r.player_id && g.loser_id === opp.id).length;
                        const losses = state.games.filter(g => g.loser_id === r.player_id && g.winner_id === opp.id).length;
                        return { opp, pred, wins, losses };
                      }).sort((a, b) => b.pred.p_a_wins - a.pred.p_a_wins);
                      return (
                        <div className="panel fade-in" style={{ padding: "12px 16px", borderTop: "none", marginTop: -1 }}>
                          <div className="section-label" style={{ marginBottom: 8, color: "var(--accent)" }}>{r.display_name.toUpperCase()} — MATCHUP PROFILE</div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            {rows.map(({ opp, pred, wins, losses }) => {
                              const p = pred.p_a_wins;
                              const color = p > 0.6 ? "var(--win)" : p < 0.4 ? "var(--lose)" : "var(--neutral)";
                              const hasGames = wins + losses > 0;
                              return (
                                <div key={opp.id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                  <span className="player-name" style={{ minWidth: 80 }}>{opp.display_name}</span>
                                  <div style={{ flex: 1, height: 4, background: "var(--surface2)", borderRadius: 2, overflow: "hidden" }}>
                                    <div style={{ height: "100%", width: pct(p), background: color, borderRadius: 2, transition: "width 0.3s" }} />
                                  </div>
                                  <span className="font-mono" style={{ fontSize: "0.84rem", color, minWidth: 36, textAlign: "right" }}>{pct(p, 0)}</span>
                                  {(() => {
                                    const { aKills, bKills } = predictScore(state, r.player_id, opp.id, p);
                                    return (
                                      <span className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-dim)", minWidth: 36, textAlign: "right" }}>
                                        <span style={{ color: p >= 0.5 ? "var(--text-bright)" : "var(--text-dim)" }}>{aKills}</span>
                                        <span style={{ color: "var(--text-dim)" }}>–</span>
                                        <span style={{ color: p < 0.5 ? "var(--text-bright)" : "var(--text-dim)" }}>{bKills}</span>
                                      </span>
                                    );
                                  })()}
                                  {hasGames && (
                                    <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)", minWidth: 40 }}>
                                      <span style={{ color: "var(--win)" }}>{wins}W</span>–<span style={{ color: "var(--lose)" }}>{losses}L</span>
                                    </span>
                                  )}
                                  {!hasGames && <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", minWidth: 40 }}>no games</span>}
                                </div>
                              );
                            })}
                          </div>
                          {(() => {
                            const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
                            let correct = 0, total = 0;
                            for (let i = 1; i < sorted.length; i++) {
                              const g = sorted[i];
                              if (g.winner_id !== r.player_id && g.loser_id !== r.player_id) continue;
                              if (!state.players[g.winner_id] || !state.players[g.loser_id]) continue;
                              const hist = { players: state.players, games: sorted.slice(0, i) };
                              const { p_a_wins } = predictPairwise(hist, g.winner_id, g.loser_id);
                              if (p_a_wins > 0.5) correct++;
                              total++;
                            }
                            if (total < 2) return null;
                            const rate = correct / total;
                            const rateColor = rate > 0.65 ? "var(--win)" : rate < 0.45 ? "var(--accent2)" : "var(--neutral)";
                            return (
                              <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--border)" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                  <span className="section-label">MODEL ACCURACY ON YOUR GAMES</span>
                                  <span className="font-mono" style={{ fontSize: "0.82rem", color: rateColor }}>{Math.round(rate * 100)}%</span>
                                  <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>({correct}/{total})</span>
                                </div>
                                <div style={{ height: 3, background: "var(--surface2)", borderRadius: 2, overflow: "hidden" }}>
                                  <div style={{ height: "100%", width: `${Math.round(rate * 100)}%`, background: rateColor, borderRadius: 2 }} />
                                </div>
                                {rate < 0.45 && <div className="font-mono" style={{ fontSize: "0.70rem", color: "var(--accent2)", marginTop: 4 }}>model consistently underestimates this player</div>}
                                {rate > 0.75 && <div className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)", marginTop: 4 }}>model reads this player well</div>}
                              </div>
                            );
                          })()}
                          {/* PageRank + Reliability */}
                          {(() => {
                            const prScore = r.page_rank;
                            const prRank = [...ranking].sort((a, b) => b.page_rank - a.page_rank).findIndex(x => x.player_id === r.player_id);
                            const rel = computeReliability(r.player_id, state.games);
                            const relLabel = rel < 0.3 ? "LOW" : rel < 0.65 ? "MED" : "HIGH";
                            const relColor = rel < 0.3 ? "var(--lose)" : rel < 0.65 ? "var(--neutral)" : "var(--win)";
                            const bt = btResult?.ratings[r.player_id];
                            const btSe = btResult?.se[r.player_id];
                            const btRank = btResult ? [...Object.entries(btResult.ratings)].sort((a, b) => b[1] - a[1]).findIndex(([id]) => id === r.player_id) : -1;
                            const eloRankIdx = [...sortedRanking].sort((a, b) => (elo[b.player_id] ?? 1000) - (elo[a.player_id] ?? 1000)).findIndex(x => x.player_id === r.player_id);
                            const btDiverges = btRank >= 0 && Math.abs(btRank - eloRankIdx) >= 2;
                            return (
                              <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--border)", display: "flex", gap: 24, flexWrap: "wrap" }}>
                                <div>
                                  <div className="section-label" style={{ marginBottom: 3 }}>PAGERANK AUTHORITY</div>
                                  <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                                    <span className="font-mono" style={{ fontSize: "0.90rem", color: "var(--accent)" }}>{(prScore * 100).toFixed(2)}</span>
                                    <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)" }}>#{prRank + 1} authority</span>
                                  </div>
                                </div>
                                <div>
                                  <div className="section-label" style={{ marginBottom: 3 }}>DATA RELIABILITY</div>
                                  <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                                    <span className="font-mono" style={{ fontSize: "0.90rem", color: relColor }}>{relLabel}</span>
                                    <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)" }}>{Math.round(rel * 100)}%</span>
                                  </div>
                                </div>
                                {bt !== undefined && btSe !== undefined && (
                                  <div>
                                    <div className="section-label" style={{ marginBottom: 3 }}>BRADLEY-TERRY</div>
                                    <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                                      <span className="font-mono" style={{ fontSize: "0.90rem", color: "var(--accent)" }}>{bt}</span>
                                      <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)" }}>±{Math.round(btSe)}</span>
                                      {btDiverges && <span className="font-mono" style={{ fontSize: "0.70rem", color: "var(--neutral)", border: "1px solid var(--neutral)", padding: "0 3px" }}>DIVERGENT</span>}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })()}
                          {/* Ceiling estimate */}
                          {(() => {
                            const rel = computeReliability(r.player_id, state.games);
                            if (rel < 0.65) return null;
                            const h = eloHistory[r.player_id] ?? [];
                            if (h.length < 8) return null;
                            const est = computeCeilingEstimate(h);
                            if (!est) return null;
                            return (
                              <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border)" }}>
                                <div className="section-label" style={{ marginBottom: 4 }}>CEILING ESTIMATE</div>
                                <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                                  <span className="font-mono" style={{ fontSize: "0.88rem", color: "var(--accent)" }}>{est.ceiling}</span>
                                  <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)" }}>±{est.se}</span>
                                  <span className="font-mono" style={{ fontSize: "0.72rem", color: est.plateaued ? "var(--neutral)" : "var(--win)" }}>
                                    {est.plateaued ? "PLATEAUED" : `+${est.gap_to_ceiling} gap`}
                                  </span>
                                </div>
                              </div>
                            );
                          })()}
                          {/* Nemesis / victim */}
                          {(() => {
                            const profile = computeNemesisProfile(state, r.player_id);
                            if (!profile.nemesis && !profile.victim) return null;
                            return (
                              <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border)", display: "flex", gap: 24, flexWrap: "wrap" }}>
                                {profile.nemesis && (
                                  <div>
                                    <div className="section-label" style={{ marginBottom: 3, color: "var(--lose)" }}>NEMESIS</div>
                                    <span className="font-mono" style={{ fontSize: "0.84rem", color: "var(--lose)" }}>
                                      {state.players[profile.nemesis.opponent_id]?.display_name ?? profile.nemesis.opponent_id}
                                    </span>
                                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>
                                      {Math.round(profile.nemesis.actual_wr * 100)}% actual vs {Math.round(profile.nemesis.expected_wr * 100)}% expected
                                    </span>
                                  </div>
                                )}
                                {profile.victim && (
                                  <div>
                                    <div className="section-label" style={{ marginBottom: 3, color: "var(--win)" }}>VICTIM</div>
                                    <span className="font-mono" style={{ fontSize: "0.84rem", color: "var(--win)" }}>
                                      {state.players[profile.victim.opponent_id]?.display_name ?? profile.victim.opponent_id}
                                    </span>
                                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>
                                      {Math.round(profile.victim.actual_wr * 100)}% actual vs {Math.round(profile.victim.expected_wr * 100)}% expected
                                    </span>
                                  </div>
                                )}
                              </div>
                            );
                          })()}
                        </div>
                      );
                    })()}
                  </div>
                );
              })}

              {/* Path distribution */}
              {Object.keys(eloHistory).some(id => eloHistory[id].length > 0) && (
                <div style={{ marginTop: 28, marginBottom: 28 }}>
                  <EloChart history={eloHistory} players={state.players} />
                </div>
              )}

              {players.length >= 2 && state.games.length > 0 && (
                <div style={{ marginTop: 28, marginBottom: 28 }}>
                  <GraphViz state={state} elo={elo} />
                </div>
              )}

              {/* Rivalry board */}
              {rivalries.length > 0 && (
                <div style={{ marginTop: 28, marginBottom: 28 }}>
                  <div className="section-label" style={{ marginBottom: 10 }}>TOP RIVALRIES</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {rivalries.map(rv => {
                      const key = `${rv.a}_${rv.b}`;
                      const aName = state.players[rv.a]?.display_name ?? rv.a;
                      const bName = state.players[rv.b]?.display_name ?? rv.b;
                      const dominance = Math.abs(rv.aWins - rv.bWins) / rv.total;
                      const color = dominance < 0.15 ? "var(--neutral)" : rv.aWins > rv.bWins ? "var(--win)" : "var(--lose)";
                      const leader = rv.aWins >= rv.bWins ? aName : bName;
                      const isOpen = selectedRivalry === key;
                      return (
                        <div key={key} style={{ marginBottom: 1 }}>
                          <div className="panel" onClick={() => setSelectedRivalry(isOpen ? null : key)}
                            style={{ padding: "9px 14px", display: "flex", alignItems: "center", gap: 12, cursor: "pointer", borderColor: isOpen ? "var(--accent)" : undefined }}>
                            <span className="player-name" style={{ minWidth: 70 }}>{aName}</span>
                            <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                              <span className="font-mono" style={{ fontSize: "0.92rem", color: "var(--win)" }}>{rv.aWins}</span>
                              <div style={{ flex: 1, height: 4, background: "var(--surface2)", borderRadius: 2, overflow: "hidden" }}>
                                <div style={{ height: "100%", width: `${(rv.aWins / rv.total) * 100}%`, background: "var(--accent)", borderRadius: 2 }} />
                              </div>
                              <span className="font-mono" style={{ fontSize: "0.92rem", color: "var(--lose)" }}>{rv.bWins}</span>
                            </div>
                            <span className="player-name" style={{ minWidth: 70, textAlign: "right" }}>{bName}</span>
                            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", minWidth: 60 }}>
                              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>{rv.total} GAMES</span>
                              <span className="font-mono" style={{ fontSize: "0.72rem", color }}>{dominance < 0.15 ? "EVEN" : `${leader.toUpperCase()} LEADS`}</span>
                            </div>
                          </div>
                          {isOpen && <RivalryTimeline rv={rv} games={state.games} players={state.players} />}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Player scatter plot */}
              {ranking.length >= 2 && (
                <div style={{ marginTop: 28, marginBottom: 28 }}>
                  <PlayerScatterPlot ranking={sortedRanking} />
                </div>
              )}

              {/* Matchup matrix */}
              {ranking.length >= 2 && (
                <div style={{ marginTop: 28 }}>
                  <div className="section-label" style={{ marginBottom: 10 }}>PAIRWISE WIN PROBABILITY MATRIX (row beats col)</div>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ borderCollapse: "collapse", fontSize: "0.82rem", fontFamily: "Share Tech Mono" }}>
                      <thead>
                        <tr>
                          <td style={{ padding: "4px 10px", color: "var(--text-dim)" }}></td>
                          {sortedRanking.map(r => (
                            <td key={r.player_id} style={{ padding: "4px 8px", color: "var(--text-dim)", textAlign: "center", maxWidth: 70, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {r.display_name.slice(0, 6)}
                            </td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sortedRanking.map(rowP => (
                          <tr key={rowP.player_id}>
                            <td style={{ padding: "4px 10px", color: "var(--text-dim)", whiteSpace: "nowrap" }}>{rowP.display_name.slice(0, 8)}</td>
                            {sortedRanking.map(colP => {
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
            <div style={{ color: "var(--neutral)", fontSize: "0.90rem", marginBottom: 14, fontFamily: "Share Tech Mono" }}>⚠ Add at least 2 players first</div>
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

            {/* Score inputs */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 36px 1fr", gap: 12, alignItems: "end", marginBottom: 14 }}>
              <div>
                <div style={{ fontSize: "0.70rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", letterSpacing: "0.1em", marginBottom: 2 }}>WINNER SCORE</div>
                <input className="input" type="number" min="0" max="99" value={scoreWinner}
                  onChange={e => setScoreWinner(e.target.value)} placeholder="e.g. 7" style={{ padding: "4px 8px", fontSize: "0.9rem" }} />
              </div>
              <div style={{ textAlign: "center", fontFamily: "Rajdhani", fontWeight: 700, color: "var(--text-dim)", paddingBottom: 10 }}>—</div>
              <div>
                <div style={{ fontSize: "0.70rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", letterSpacing: "0.1em", marginBottom: 2 }}>LOSER SCORE</div>
                <input className="input" type="number" min="0" max="99" value={scoreLoser}
                  onChange={e => setScoreLoser(e.target.value)} placeholder="e.g. 3" style={{ padding: "4px 8px", fontSize: "0.9rem" }} />
              </div>
            </div>

            {/* Stats inputs */}
            <div className="panel" style={{ padding: 16, marginBottom: 14 }}>
              <div style={{ display: "flex", gap: 20 }}>
                <StatInput label={logWinner ? `${state.players[logWinner]?.display_name ?? "WINNER"} STATS` : "WINNER STATS"} stats={winStats}
                  onChange={s => { setWinStats(s); setLoseStats({ kills: s.deaths, deaths: s.kills }); }} />
                <div style={{ width: 1, background: "var(--border)", flexShrink: 0 }} />
                <StatInput label={logLoser ? `${state.players[logLoser]?.display_name ?? "LOSER"} STATS` : "LOSER STATS"} stats={loseStats}
                  onChange={s => setLoseStats(s)} />
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button type="submit" className="btn btn-primary" disabled={logging || !logWinner || !logLoser || players.length < 2}>
                {logging ? "LOGGING..." : "LOG GAME"}
              </button>
              {logMsg && <span className="font-mono" style={{ color: "var(--win)", fontSize: "0.86rem" }}>{logMsg}</span>}
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
            <button className="btn btn-primary" onClick={getPrediction} disabled={!predA || !predB} style={{ width: "100%" }}>
              PREDICT
            </button>
          </div>

          {predA && players.length > 1 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <div className="section-label">SUGGESTED OPPONENTS FOR {state.players[predA]?.display_name?.toUpperCase()}</div>
                <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
                  {(["closest", "elovalue", "uncertain"] as const).map(s => (
                    <button key={s} className="btn" onClick={() => setSuggestSort(s)}
                      style={{ padding: "1px 7px", fontSize: "0.70rem", background: suggestSort === s ? "var(--accent)" : undefined, color: suggestSort === s ? "#000" : undefined }}>
                      {s === "closest" ? "CLOSEST" : s === "elovalue" ? "ELO VALUE" : "UNCERTAIN"}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {players
                  .filter(p => p.id !== predA)
                  .map(p => {
                    const pred = predictPairwise(state, predA, p.id);
                    const diff = Math.abs(pred.p_a_wins - 0.5);
                    const eloMap = elo;
                    const rA = eloMap[predA] ?? 1000, rB = eloMap[p.id] ?? 1000;
                    const eloExpected = 1 / (1 + Math.pow(10, (rB - rA) / 400));
                    // ELO value = expected gain for underdog (pred favors them more than elo)
                    const eloValue = pred.p_a_wins > eloExpected ? pred.p_a_wins - eloExpected : eloExpected - pred.p_a_wins;
                    const uncertain = 1 - pred.confidence;
                    const sortScore = suggestSort === "closest" ? -diff : suggestSort === "elovalue" ? eloValue : uncertain;
                    return { player: p, pred, diff, sortScore };
                  })
                  .sort((a, b) => b.sortScore - a.sortScore)
                  .map(({ player, pred }) => {
                    const color = pred.p_a_wins > 0.6 ? "var(--win)" : pred.p_a_wins < 0.4 ? "var(--lose)" : "var(--neutral)";
                    const urgency = computeRematchUrgency(state, predA, player.id);
                    const alert = computeUpsetAlert(predA, player.id, pred.p_a_wins, formRating);
                    return (
                      <div key={player.id} className="panel" style={{ padding: "8px 12px", display: "flex", alignItems: "center", gap: 8, cursor: "pointer", borderColor: alert ? "var(--neutral)" : undefined }}
                        onClick={() => { setPredB(player.id); setPrediction(pred); }}>
                        <span className="player-name" style={{ flex: 1 }}>{player.display_name}</span>
                        {urgency && urgency.prediction_shift >= 0.1 && (
                          <span className="font-mono" style={{ fontSize: "0.67rem", color: "var(--neutral)", border: "1px solid var(--neutral)", padding: "0 3px" }}>
                            REMATCH ↑{Math.round(urgency.prediction_shift * 100)}%
                          </span>
                        )}
                        {alert && (
                          <span className="font-mono" style={{ fontSize: "0.67rem", color: "var(--neutral)", border: "1px solid var(--neutral)", padding: "0 3px" }}>
                            ⚡ {alert.alert_level}
                          </span>
                        )}
                        <span className="font-mono" style={{ fontSize: "0.84rem", color, marginRight: 4 }}>{pct(pred.p_a_wins, 1)}</span>
                        <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                          {pred.p_a_wins > 0.6 ? "FAVORED" : pred.p_a_wins < 0.4 ? "UNDERDOG" : "EVEN"}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {prediction && (() => {
            const aName = state.players[predA]?.display_name ?? "A";
            const bName = state.players[predB]?.display_name ?? "B";
            const simWins = h2hSim[predA]?.[predB];
            const pA = simWins !== undefined ? simWins / 1000 : prediction.p_a_wins;
            const pB = 1 - pA;
            const prior = prediction.p_a_wins;
            const priorDelta = Math.abs(pA - prior);
            if (priorDelta > 0.05) console.warn(`[GraphELO] MC/prior delta ${(priorDelta * 100).toFixed(1)}% — path weights diverging from simulation for this matchup.`);
            const h2h = state.games.filter(
              g => (g.winner_id === predA && g.loser_id === predB) ||
                   (g.winner_id === predB && g.loser_id === predA)
            );
            const aWins = h2h.filter(g => g.winner_id === predA).length;
            const bWins = h2h.length - aWins;
            const ciA = katzCI(pA, prediction.direct_games, prediction.evidence_mass);
            const ciB: [number, number] = [1 - ciA[1], 1 - ciA[0]];
            return (
              <div className="fade-in">
                <div className="panel panel-accent2" style={{ padding: 18 }}>
                  {/* Win probability bar */}
                  <div style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 6 }}>
                      <div>
                        <div className="player-name">{aName}</div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 5 }}>
                          <span className="rating-value" style={{ fontSize: "1.6rem" }}>{pct(pA, 1)}</span>
                          <span className="font-mono" style={{ fontSize: "0.78rem", color: "var(--text-dim)", letterSpacing: "0.02em" }}>
                            ({Math.round(ciA[0] * 100)}–{Math.round(ciA[1] * 100)}%)
                          </span>
                        </div>
                      </div>
                      {h2h.length > 0 && (
                        <div style={{ textAlign: "center", paddingBottom: 2 }}>
                          <span className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-dim)" }}>RECORD </span>
                          <span className="font-mono" style={{ fontSize: "0.92rem", color: "var(--win)" }}>{aWins}</span>
                          <span className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-dim)" }}> — </span>
                          <span className="font-mono" style={{ fontSize: "0.92rem", color: "var(--lose)" }}>{bWins}</span>
                        </div>
                      )}
                      <div style={{ textAlign: "right" }}>
                        <div className="player-name">{bName}</div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 5, justifyContent: "flex-end" }}>
                          <span className="font-mono" style={{ fontSize: "0.78rem", color: "var(--text-dim)", letterSpacing: "0.02em" }}>
                            ({Math.round(ciB[0] * 100)}–{Math.round(ciB[1] * 100)}%)
                          </span>
                          <span className="rating-value" style={{ fontSize: "1.6rem", color: pB > pA ? "var(--accent)" : "var(--lose)" }}>{pct(pB, 1)}</span>
                        </div>
                      </div>
                    </div>
                    {/* Split bar */}
                    <div style={{ height: 6, background: "var(--lose)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: pct(pA), background: "var(--accent)", transition: "width 0.5s" }} />
                    </div>
                  </div>

                  {/* Predicted score */}
                  {state.games.length >= 2 && (() => {
                    const { aKills, bKills } = predictScore(state, predA, predB, pA);
                    const aWins = pA >= 0.5;
                    return (
                      <div style={{ textAlign: "center", marginBottom: 18, paddingBottom: 18, borderBottom: "1px solid var(--border)" }}>
                        <div className="section-label" style={{ marginBottom: 8 }}>PREDICTED SCORE</div>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 14 }}>
                          <span className="player-name" style={{ color: aWins ? "var(--win)" : "var(--text-dim)", minWidth: 60, textAlign: "right" }}>{aName}</span>
                          <span className="font-display" style={{ fontSize: "2.4rem", fontWeight: 700, letterSpacing: "0.04em", color: "var(--text-bright)", lineHeight: 1 }}>
                            {aKills}<span style={{ color: "var(--text-dim)", margin: "0 6px", fontSize: "1.8rem" }}>–</span>{bKills}
                          </span>
                          <span className="player-name" style={{ color: !aWins ? "var(--win)" : "var(--text-dim)", minWidth: 60 }}>{bName}</span>
                        </div>
                      </div>
                    );
                  })()}

                  {/* Evidence breakdown */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
                    {[
                      { label: "DIRECT GAMES", val: prediction.direct_games.toString() },
                      { label: "KATZ MASS", val: fmt(prediction.evidence_mass, 3) },
                      { label: "CONFIDENCE", val: confLabel(prediction.confidence), color: confColor(prediction.confidence) },
                    ].map(({ label, val, color }) => (
                      <div key={label} style={{ padding: "8px 10px", border: "1px solid var(--border)" }}>
                        <div className="section-label" style={{ marginBottom: 4 }}>{label}</div>
                        <div className="font-mono" style={{ fontSize: "0.95rem", color: color ?? "var(--text-bright)" }}>{val}</div>
                      </div>
                    ))}
                  </div>
                  {prediction.top_contributors.length > 0 && (
                    <div style={{ marginTop: 14 }}>
                      <div className="section-label" style={{ marginBottom: 8 }}>INFERRED VIA</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        {(() => {
                          const total = prediction.top_contributors.reduce((s, c) => s + c.mass, 0);
                          return prediction.top_contributors.map(({ playerId, mass }) => {
                            const mName = state.players[playerId]?.display_name ?? playerId;
                            return (
                              <div key={playerId} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)", minWidth: 120 }}>
                                  {aName} → {mName} → {bName}
                                </span>
                                <div style={{ flex: 1, height: 5, background: "var(--surface2)", borderRadius: 2, overflow: "hidden" }}>
                                  <div style={{ height: "100%", width: `${total > 0 ? Math.round((mass / total) * 100) : 0}%`, background: "var(--accent)", borderRadius: 2 }} />
                                </div>
                                <span className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)", minWidth: 32, textAlign: "right" }}>
                                  {(mass * 100).toFixed(1)}
                                </span>
                              </div>
                            );
                          });
                        })()}
                      </div>
                    </div>
                  )}
                  {/* Upset alert */}
                  {(() => {
                    const alert = computeUpsetAlert(predA, predB, pA, formRating);
                    if (!alert) return null;
                    const underdogName = state.players[alert.underdog]?.display_name ?? alert.underdog;
                    return (
                      <div style={{ marginTop: 12, padding: "8px 10px", border: "1px solid var(--neutral)", color: "var(--neutral)" }}>
                        <div className="font-mono" style={{ fontSize: "0.73rem", marginBottom: 2 }}>
                          ⚡ UPSET ALERT ({alert.alert_level})
                        </div>
                        <div className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)" }}>
                          Katz: {pct(alert.elo_win_prob, 1)} · Form: {pct(alert.form_win_prob, 1)} · Δ{Math.round(alert.divergence * 100)}%
                        </div>
                        <div className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)" }}>
                          {underdogName}&apos;s form rating diverges from model prediction
                        </div>
                      </div>
                    );
                  })()}
                  {/* Rematch urgency */}
                  {(() => {
                    if (!predA || !predB) return null;
                    const urg = computeRematchUrgency(state, predA, predB);
                    if (!urg || urg.prediction_shift < 0.1) return null;
                    return (
                      <div style={{ marginTop: 8, padding: "6px 10px", border: "1px solid var(--border)" }}>
                        <div className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)" }}>
                          REMATCH CONTEXT · {urg.days_since}d ago · prediction shift +{Math.round(urg.prediction_shift * 100)}% · {urg.direction}
                        </div>
                      </div>
                    );
                  })()}
                  {/* Compact sim metadata — model prior shown as diagnostic input, not headline */}
                  <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--border)" }}>
                    <div className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                      <span>{simWins !== undefined ? `${simWins}/1000 runs` : "no sim data"}</span>
                      <span style={{ color: "var(--border2)" }}>·</span>
                      {h2h.length > 0
                        ? <span>direct {aWins}W–{bWins}L</span>
                        : prediction.evidence_mass > 0
                          ? <span>katz ↑{fmt(prediction.katz_ab, 3)} ↓{fmt(prediction.katz_ba, 3)}</span>
                          : <span>no graph evidence</span>}
                      <span style={{ color: "var(--border2)" }}>·</span>
                      <span>model prior: {pct(prior, 1)}</span>
                      {priorDelta > 0.005 && (
                        <>
                          <span style={{ color: "var(--border2)" }}>·</span>
                          <span style={{ color: priorDelta > 0.05 ? "var(--neutral)" : undefined }}>Δ {pct(priorDelta, 1)}</span>
                        </>
                      )}
                    </div>
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
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <div className="section-label">
              GAME HISTORY {historyFilter
                ? `· ${winsGames.length}W ${lossesGames.length}L`
                : `(${state.games.length})`}
            </div>
            <select className="input" value={historyFilter} onChange={e => setHistoryFilter(e.target.value)} style={{ maxWidth: 180, padding: "3px 8px", fontSize: "0.84rem" }}>
              <option value="">all players</option>
              {players.map(p => <option key={p.id} value={p.id}>{p.display_name}</option>)}
            </select>
            {historyFilter && <button className="btn" style={{ padding: "2px 8px", fontSize: "0.7rem" }} onClick={() => setHistoryFilter("")}>clear</button>}
          </div>

          {filteredGames.length === 0 && <div style={{ color: "var(--text-dim)", fontSize: "0.92rem" }}>{historyFilter ? "No games for this player." : "No games logged yet."}</div>}

          {(() => {
            const gameCard = (g: typeof state.games[0]) => {
              const w = state.players[g.winner_id];
              const l = state.players[g.loser_id];
              const ws = g.winner_stats;
              const ls = g.loser_stats;
              const wKD = ws.deaths > 0 ? (ws.kills / ws.deaths).toFixed(2) : ws.kills.toString();
              const lKD = ls.deaths > 0 ? (ls.kills / ls.deaths).toFixed(2) : ls.kills.toString();
              return (
                <div key={g.id} className="panel" style={{ padding: "10px 14px" }}>
                  <div style={{ display: "flex", alignItems: "center" }}>
                    <div style={{ flex: 1 }}>
                      <span className="player-name" style={{ color: "var(--win)" }}>{w?.display_name ?? g.winner_id}</span>
                      <div className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: 2 }}>
                        {ws.kills}K {ws.deaths}D · KD {wKD}
                      </div>
                    </div>
                    <div style={{ padding: "0 14px", textAlign: "center" }}>
                      <div className="font-mono" style={{ fontSize: "0.72rem", color: "var(--accent)" }}>DEF</div>
                      {upsetMap[g.id] && (
                        <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--neutral)", border: "1px solid var(--neutral)", padding: "0 3px", marginTop: 2, lineHeight: 1.5 }}>UPSET</div>
                      )}
                      {revengeMap[g.id] && (
                        <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--accent2)", border: "1px solid var(--accent2)", padding: "0 3px", marginTop: 2, lineHeight: 1.5 }}>REVENGE</div>
                      )}
                      <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", marginTop: 2 }}>
                        {new Date(g.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                    <div style={{ flex: 1, textAlign: "right" }}>
                      <span className="player-name" style={{ color: "var(--lose)" }}>{l?.display_name ?? g.loser_id}</span>
                      <div className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: 2 }}>
                        KD {lKD} · {ls.deaths}D {ls.kills}K
                      </div>
                    </div>
                    <button onClick={() => handleDeleteGame(g.id)}
                      style={{ marginLeft: 12, background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer", fontSize: "0.92rem", lineHeight: 1, padding: "2px 4px" }}
                      title="Delete game">✕</button>
                  </div>
                </div>
              );
            };

            if (historyFilter) {
              const trend = computeRecentStats(historyFilter, state.games);
              const opponentIds = Array.from(new Set(
                state.games
                  .filter(g => g.winner_id === historyFilter || g.loser_id === historyFilter)
                  .map(g => g.winner_id === historyFilter ? g.loser_id : g.winner_id)
              ));
              const nemeses = opponentIds.filter(oppId =>
                lossesGames.some(g => g.winner_id === oppId) && !winsGames.some(g => g.loser_id === oppId)
              );
              const dominates = opponentIds.filter(oppId =>
                winsGames.some(g => g.loser_id === oppId) && !lossesGames.some(g => g.winner_id === oppId)
              );
              return (
                <>
                  {trend && (() => {
                    const arrow = (r: number, o: number) => r > o + 0.05 ? { sym: "↑", col: "var(--win)" } : r < o - 0.05 ? { sym: "↓", col: "var(--lose)" } : { sym: "→", col: "var(--text-dim)" };
                    const kdA  = arrow(trend.recent.kd,       trend.overall.kd);
                    const wrA  = arrow(trend.recent.win_rate, trend.overall.win_rate);
                    return (
                      <>
                      <RollingFormChart playerId={historyFilter} games={state.games} />
                      <div className="panel" style={{ padding: "10px 14px", marginBottom: 12 }}>
                        <div className="section-label" style={{ marginBottom: 8 }}>RECENT FORM — last 5 vs all-time</div>
                        <div style={{ display: "flex", gap: 20 }}>
                          {[
                            { label: "KD",   r: trend.recent.kd,       o: trend.overall.kd,       a: kdA, fmt: (v: number) => v.toFixed(2) },
                            { label: "WIN%", r: trend.recent.win_rate,  o: trend.overall.win_rate, a: wrA, fmt: (v: number) => `${(v*100).toFixed(0)}%` },
                          ].map(({ label, r, o, a, fmt }) => (
                            <div key={label}>
                              <div className="section-label" style={{ marginBottom: 3 }}>{label}</div>
                              <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                                <span className="font-mono" style={{ fontSize: "0.9rem", color: "var(--text-bright)" }}>{fmt(r)}</span>
                                <span className="font-mono" style={{ fontSize: "0.7rem", color: a.col }}>{a.sym}</span>
                                <span className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>{fmt(o)}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                      </>
                    );
                  })()}
                  {(nemeses.length > 0 || dominates.length > 0) && (
                    <div className="panel" style={{ padding: "10px 14px", marginBottom: 16, display: "flex", gap: 28, flexWrap: "wrap" }}>
                      {nemeses.length > 0 && (
                        <div>
                          <div className="section-label" style={{ marginBottom: 6, color: "var(--lose)" }}>NEMESIS — never beaten</div>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            {nemeses.map(id => (
                              <span key={id} className="font-mono" style={{ fontSize: "0.82rem", color: "var(--lose)", padding: "2px 7px", border: "1px solid var(--lose)" }}>
                                {state.players[id]?.display_name ?? id}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {dominates.length > 0 && (
                        <div>
                          <div className="section-label" style={{ marginBottom: 6, color: "var(--win)" }}>DOMINATES — never lost to</div>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            {dominates.map(id => (
                              <span key={id} className="font-mono" style={{ fontSize: "0.82rem", color: "var(--win)", padding: "2px 7px", border: "1px solid var(--win)" }}>
                                {state.players[id]?.display_name ?? id}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  {winsGames.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <div className="section-label" style={{ marginBottom: 6, color: "var(--win)" }}>WINS ({winsGames.length})</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>{winsGames.map(gameCard)}</div>
                    </div>
                  )}
                  {lossesGames.length > 0 && (
                    <div>
                      <div className="section-label" style={{ marginBottom: 6, color: "var(--lose)" }}>LOSSES ({lossesGames.length})</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>{lossesGames.map(gameCard)}</div>
                    </div>
                  )}
                </>
              );
            }

            // Group into sessions: gap > 2h starts a new session
            const SESSION_GAP = 2 * 60 * 60 * 1000;
            const sessions: (typeof filteredGames)[] = [];
            let cur: typeof filteredGames = [];
            for (const g of filteredGames) {
              if (cur.length === 0 || Math.abs(cur[cur.length - 1].timestamp - g.timestamp) <= SESSION_GAP) {
                cur.push(g);
              } else {
                sessions.push(cur);
                cur = [g];
              }
            }
            if (cur.length > 0) sessions.push(cur);

            return (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {sessions.map((session, si) => {
                  const d = new Date(session[0].timestamp);
                  const label = d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }).toUpperCase();
                  return (
                    <div key={si}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                        <span className="section-label">{label}</span>
                        <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>·  {session.length} {session.length === 1 ? "GAME" : "GAMES"}</span>
                        <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        {session.map(gameCard)}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}
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
            {players.length === 0 && <div style={{ color: "var(--text-dim)", fontSize: "0.90rem" }}>No players yet.</div>}
            {players.map(p => {
              const r = ranking.find(x => x.player_id === p.id);
              const sv = r?.stat_vec;
              const wins = state.games.filter(g => g.winner_id === p.id).length;
              const losses = state.games.filter(g => g.loser_id === p.id).length;
              return (
                <div key={p.id} className="panel" style={{ padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div className="player-name">{p.display_name}</div>
                    {sv && sv.games_played > 0 && (
                      <div className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)", marginTop: 2 }}>
                        KD {fmt(sv.kd)} · KPR {fmt(sv.kpr, 1)} · <span style={{ color: "var(--win)" }}>{wins}W</span>–<span style={{ color: "var(--lose)" }}>{losses}L</span>
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    {sv && sv.games_played > 0 && (
                      <button className="btn" style={{ padding: "2px 8px", fontSize: "0.75rem" }}
                        onClick={() => { setHistoryFilter(p.id); setTab("history"); }}>
                        HISTORY
                      </button>
                    )}
                    {r && (
                      <div style={{ textAlign: "right" }}>
                        <div className="rating-value" style={{ fontSize: "0.9rem" }}>{pct(r.tournament_win_pct, 1)}</div>
                        <div style={{ fontSize: "0.72rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono" }}>CHAMP%</div>
                      </div>
                    )}
                    <button onClick={() => handleDeletePlayer(p.id)}
                      style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer", fontSize: "0.92rem", padding: "2px 4px" }}
                      title="Delete player">✕</button>
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: 16, padding: "10px 12px", border: "1px dashed var(--border)", fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "Share Tech Mono", lineHeight: 1.6 }}>
            API HOOK READY — swap computeStatVector() in lib/graph-engine.ts<br />
            with r6data.com fetch to pull ranked stats automatically.
          </div>
        </div>
      )}

      {/* ── SYSTEM ── */}
      {tab === "system" && (
        <div className="fade-in">
          <div className="section-label" style={{ marginBottom: 16 }}>SYSTEM ANALYTICS</div>

          {/* Model Validation */}
          <div className="panel" style={{ padding: "14px 18px", marginBottom: 14 }}>
            <div className="section-label" style={{ marginBottom: 10, color: "var(--accent)" }}>MODEL VALIDATION</div>
            {looCv ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>LOO ACCURACY</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: looCv.accuracy > 0.6 ? "var(--win)" : "var(--text-bright)" }}>{Math.round(looCv.accuracy * 100)}%</span>
                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>({looCv.n} predictions)</span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>LOO BRIER</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: "var(--text-bright)" }}>{looCv.brier_score.toFixed(3)}</span>
                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>random=0.250</span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>BRIER SKILL</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: looCv.brier_skill > 0.05 ? "var(--win)" : "var(--lose)" }}>+{Math.round(looCv.brier_skill * 100)}%</span>
                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>vs random</span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>IN-SAMPLE</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: "var(--text-dim)" }}>{Math.round(looCv.in_sample_accuracy * 100)}%</span>
                    <span className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)", marginLeft: 6 }}>← optimistic</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-dim)" }}>Need ≥4 games for LOO validation.</div>
            )}
          </div>

          {/* Bradley-Terry vs ELO vs Katz */}
          {btResult && ranking.length >= 2 && (() => {
            const eloOrder = [...ranking].sort((a, b) => (elo[b.player_id] ?? 1000) - (elo[a.player_id] ?? 1000)).map(r => r.player_id);
            const katzOrder = ranking.map(r => r.player_id);
            const btOrder = [...btResult.ids].sort((a, b) => (btResult.ratings[b] ?? 1000) - (btResult.ratings[a] ?? 1000));
            const katzVsElo  = kendallTauAgreement(katzOrder, eloOrder);
            const katzVsBt   = kendallTauAgreement(katzOrder, btOrder);
            const eloVsBt    = kendallTauAgreement(eloOrder, btOrder);
            return (
              <div className="panel" style={{ padding: "14px 18px", marginBottom: 14 }}>
                <div className="section-label" style={{ marginBottom: 10, color: "var(--accent)" }}>RANKING AGREEMENT (KENDALL τ)</div>
                <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 10 }}>
                  {[
                    { label: "KATZ vs ELO",  tau: katzVsElo },
                    { label: "KATZ vs BT",   tau: katzVsBt },
                    { label: "ELO vs BT",    tau: eloVsBt },
                  ].map(({ label, tau }) => (
                    <div key={label}>
                      <div className="section-label" style={{ marginBottom: 2 }}>{label}</div>
                      <span className="font-mono" style={{ fontSize: "0.9rem", color: tau > 0.8 ? "var(--win)" : tau > 0.5 ? "var(--neutral)" : "var(--lose)" }}>{tau.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
                <div className="section-label" style={{ marginBottom: 6 }}>BT RATINGS</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {btOrder.map((id, i) => {
                    const btRating = btResult.ratings[id] ?? 1000;
                    const se = btResult.se[id] ?? 0;
                    const eloRankIdx = eloOrder.indexOf(id);
                    const diff = Math.abs(i - eloRankIdx);
                    return (
                      <div key={id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span className="font-mono" style={{ fontSize: "0.73rem", color: "var(--text-dim)", minWidth: 18 }}>#{i + 1}</span>
                        <span className="player-name" style={{ minWidth: 100 }}>{state.players[id]?.display_name ?? id}</span>
                        <span className="font-mono" style={{ fontSize: "0.90rem", color: "var(--accent)" }}>{btRating}</span>
                        <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>±{Math.round(se)}</span>
                        {diff >= 2 && (
                          <span className="font-mono" style={{ fontSize: "0.67rem", color: "var(--neutral)", border: "1px solid var(--neutral)", padding: "0 3px" }}>
                            ELO #{eloRankIdx + 1}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })()}

          {/* Parity + skill gap */}
          {state.games.length >= 4 && (() => {
            const parity = computeParityIndex(state, champCounts, 1000);
            return (
              <div className="panel" style={{ padding: "14px 18px", marginBottom: 14 }}>
                <div className="section-label" style={{ marginBottom: 10, color: "var(--accent)" }}>POOL PARITY</div>
                <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>PARITY INDEX</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: parity.normalized_parity > 0.7 ? "var(--win)" : parity.normalized_parity > 0.4 ? "var(--neutral)" : "var(--lose)" }}>
                      {Math.round(parity.normalized_parity * 100)}%
                    </span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>ELO SPREAD</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: "var(--text-bright)" }}>±{Math.round(parity.elo_spread)}</span>
                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>pts std dev</span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>COMPETITIVE ENTROPY</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: "var(--text-bright)" }}>{parity.entropy.toFixed(2)}</span>
                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>/ {parity.max_entropy.toFixed(2)} max</span>
                  </div>
                  {skillGap && (
                    <div>
                      <div className="section-label" style={{ marginBottom: 2 }}>SKILL GAP TREND</div>
                      <span className="font-mono" style={{ fontSize: "0.9rem", color: skillGap.trend === "CONVERGING" ? "var(--win)" : skillGap.trend === "DIVERGING" ? "var(--lose)" : "var(--text-dim)" }}>
                        {skillGap.trend}
                      </span>
                      <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>
                        {skillGap.slope > 0 ? "+" : ""}{skillGap.slope} pts/game
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}

          {/* 3-Cycle Detection */}
          {cycleAnalysis && (() => {
            return (
              <div className="panel" style={{ padding: "14px 18px", marginBottom: 14 }}>
                <div className="section-label" style={{ marginBottom: 10, color: "var(--accent)" }}>CYCLE ANALYSIS</div>
                <div style={{ display: "flex", gap: 24, marginBottom: 10, flexWrap: "wrap" }}>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>TRANSITIVITY RATE</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: cycleAnalysis.transitivity_rate > 0.7 ? "var(--win)" : "var(--neutral)" }}>
                      {Math.round(cycleAnalysis.transitivity_rate * 100)}%
                    </span>
                    <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: 6 }}>of matchups follow A beats B beats C → A beats C</span>
                  </div>
                  <div>
                    <div className="section-label" style={{ marginBottom: 2 }}>3-CYCLES FOUND</div>
                    <span className="font-mono" style={{ fontSize: "0.9rem", color: cycleAnalysis.cycles.length > 0 ? "var(--neutral)" : "var(--win)" }}>
                      {cycleAnalysis.cycles.length}
                    </span>
                  </div>
                </div>
                {cycleAnalysis.cycles.length > 0 && (
                  <>
                    <div className="section-label" style={{ marginBottom: 6 }}>RIVALRY TRIANGLES</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      {cycleAnalysis.cycles.slice(0, 5).map((cy, idx) => (
                        <div key={idx} className="panel" style={{ padding: "6px 10px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            {cy.players.map((id, i) => (
                              <span key={id} className="font-mono" style={{ fontSize: "0.78rem", color: "var(--text-bright)" }}>
                                {state.players[id]?.display_name ?? id}{i < 2 ? " →" : ""}
                              </span>
                            ))}
                            <span className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)", marginLeft: "auto" }}>
                              → {cy.players[0] ? (state.players[cy.players[0]]?.display_name ?? cy.players[0]) : ""} · strength {cy.strength.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
                {cycleAnalysis.total_triples === 0 && (
                  <div className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-dim)" }}>
                    Need ≥2 games per pair to detect cycles.
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}

      <div style={{ marginTop: 36, paddingTop: 14, borderTop: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 8 }}>
        <div className="font-mono" style={{ fontSize: "0.70rem", color: "var(--text-dim)", letterSpacing: "0.1em" }}>
          GRAPHELO v3 · KATZ GRAPH DIFFUSION β=0.5 · τ=90d · 1000-TOURNAMENT MONTE CARLO
        </div>
        <a href="https://eloboard.vercel.app/" target="_blank" rel="noopener noreferrer"
          className="font-mono" style={{ fontSize: "0.70rem", color: "var(--accent)", letterSpacing: "0.1em", textDecoration: "none", opacity: 0.8 }}>
          ELOBOARD ↗
        </a>
      </div>
    </div>
  );
}

