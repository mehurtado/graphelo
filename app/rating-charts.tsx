import { useState } from "react";
import { ELO_START } from "@/lib/graph-engine";
import type { EloHistoryPoint, Game } from "@/lib/graph-engine";

export const PLAYER_COLORS = ["#f59e0b","#34d399","#f87171","#a78bfa","#38bdf8","#fb923c","#e879f9","#4ade80","#f472b6","#60a5fa"];

export function EloChart({
  history, players,
}: {
  history: Record<string, EloHistoryPoint[]>;
  players: Record<string, { display_name: string }>;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<{ x: number; y: number; id: string; pt: EloHistoryPoint } | null>(null);

  const ids = Object.keys(history).filter(id => history[id].length > 0);
  if (ids.length === 0) return null;
  const visibleIds = ids.filter(id => !hidden.has(id));

  const allPts = ids.flatMap(id => history[id]);
  const minT = Math.min(...allPts.map(p => p.timestamp));
  const maxT = Math.max(...allPts.map(p => p.timestamp));

  // Y axis fits only the visible players, falling back to everyone if all are hidden.
  const scaleIds = visibleIds.length > 0 ? visibleIds : ids;
  const scalePts = scaleIds.flatMap(id => history[id]);
  const rawMin = Math.min(ELO_START, ...scalePts.map(p => p.elo));
  const rawMax = Math.max(ELO_START, ...scalePts.map(p => p.elo));
  const ePad = Math.max(15, Math.ceil((rawMax - rawMin) * 0.15));
  const minElo = rawMin - ePad, maxElo = rawMax + ePad;
  const W = 760, H = 180, PL = 44, PR = 92, PT = 10, PB = 18;
  const cw = W - PL - PR, ch = H - PT - PB;
  const xS = (t: number) => PL + (maxT === minT ? cw / 2 : ((t - minT) / (maxT - minT)) * cw);
  const yS = (e: number) => PT + ch - ((e - minElo) / (maxElo - minElo)) * ch;
  const ticks = Array.from({ length: 5 }, (_, i) => Math.round(minElo + (i / 4) * (maxElo - minElo)));

  function toggle(id: string) {
    setHidden(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
    setHover(null);
  }

  return (
    <div>
      <div style={{ position: "relative" }}>
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
            if (hidden.has(id)) return null;
            const pts = history[id];
            const color = PLAYER_COLORS[ci % PLAYER_COLORS.length];
            const draw = [{ timestamp: minT, elo: ELO_START }, ...pts];
            const d = draw.map((p, i) => `${i === 0 ? "M" : "L"}${xS(p.timestamp).toFixed(1)},${yS(p.elo).toFixed(1)}`).join(" ");
            const last = draw[draw.length - 1];
            return (
              <g key={id}>
                <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" opacity={0.85} />
                {pts.map((p, pi) => (
                  <circle
                    key={pi}
                    cx={xS(p.timestamp)} cy={yS(p.elo)} r={2.5}
                    fill="var(--surface)" stroke={color} strokeWidth={1.5}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover({ x: xS(p.timestamp), y: yS(p.elo), id, pt: p })}
                    onMouseLeave={() => setHover(null)}
                  />
                ))}
                <circle cx={xS(last.timestamp)} cy={yS(last.elo)} r={3} fill={color} />
                <text x={xS(last.timestamp) + 6} y={yS(last.elo) + 4} fill={color} fontSize={9} fontFamily="Share Tech Mono">
                  {(players[id]?.display_name ?? id).slice(0, 10)}
                </text>
              </g>
            );
          })}
        </svg>
        {hover && (() => {
          const opp = players[hover.pt.opponent_id]?.display_name ?? hover.pt.opponent_id;
          const won = hover.pt.result === "W";
          const score = hover.pt.score_for !== undefined && hover.pt.score_against !== undefined
            ? `${hover.pt.score_for}-${hover.pt.score_against}` : null;
          const color = PLAYER_COLORS[ids.indexOf(hover.id) % PLAYER_COLORS.length];
          return (
            <div style={{
              position: "absolute",
              left: `${(hover.x / W) * 100}%`, top: `${(hover.y / H) * 100}%`,
              transform: "translate(-50%, calc(-100% - 10px))",
              background: "color-mix(in srgb, var(--bg) 96%, var(--accent))",
              border: "1px solid var(--border2)", borderRadius: 5,
              padding: "6px 9px", fontSize: "0.74rem", fontFamily: "Share Tech Mono",
              whiteSpace: "nowrap", pointerEvents: "none", zIndex: 50,
              boxShadow: "0 6px 20px rgba(0, 0, 0, 0.45)",
            }}>
              <div style={{ color: "var(--text-bright)", marginBottom: 2 }}>
                {(players[hover.id]?.display_name ?? hover.id)} vs {opp}
              </div>
              <div style={{ color: won ? "var(--win)" : "var(--lose)" }}>
                {won ? "WIN" : "LOSS"}{score ? ` ${score}` : ""}
                <span style={{ color: "var(--text-dim)" }}> · </span>
                <span style={{ color }}>{hover.pt.delta >= 0 ? "+" : ""}{hover.pt.delta} ELO</span>
              </div>
            </div>
          );
        })()}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px", marginTop: 8, paddingLeft: PL }}>
        {ids.map((id, ci) => {
          const color = PLAYER_COLORS[ci % PLAYER_COLORS.length];
          const isHidden = hidden.has(id);
          return (
            <button
              key={id}
              onClick={() => toggle(id)}
              title={isHidden ? "Click to show" : "Click to hide"}
              style={{
                display: "flex", alignItems: "center", gap: 5,
                background: "none", border: "none", padding: "2px 0", cursor: "pointer",
                fontFamily: "Share Tech Mono", fontSize: "0.74rem",
                color: isHidden ? "var(--text-dim)" : color,
                opacity: isHidden ? 0.5 : 1,
                textDecoration: isHidden ? "line-through" : "none",
              }}
            >
              <span style={{ width: 9, height: 9, borderRadius: "50%", background: color, opacity: isHidden ? 0.4 : 1, flexShrink: 0 }} />
              {players[id]?.display_name ?? id}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function GraphViz({ state, elo }: { state: { players: Record<string, { display_name: string }>; games: Game[] }; elo: Record<string, number> }) {
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
              <span className="font-mono" style={{ fontSize: "0.80rem", color: "var(--win)" }}>
                APEX — {(state.players[id] as { display_name: string })?.display_name} has beaten everyone they&apos;ve faced
              </span>
            </div>
          ))}
          {sinks.map(id => (
            <div key={id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <svg width={12} height={12}><circle cx={6} cy={6} r={5} fill="none" stroke="var(--lose)" strokeWidth={1.5} strokeDasharray="3 1.5" /></svg>
              <span className="font-mono" style={{ fontSize: "0.80rem", color: "var(--lose)" }}>
                WINLESS — {(state.players[id] as { display_name: string })?.display_name} has lost to everyone they&apos;ve faced
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function RollingFormChart({ playerId, games }: { playerId: string; games: Game[] }) {
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

export function RivalryTimeline({ rv, games, players }: {
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
              fontSize: "0.76rem", fontFamily: "Share Tech Mono", color: "#000", fontWeight: 700,
            }}>
              {(aWon ? aName : bName).slice(0, 3).toUpperCase()}
            </div>
            <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>{aScore}–{bScore}</span>
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
              <span className="font-mono" style={{ fontSize: "0.76rem", color: "var(--text-dim)", minWidth: 18 }}>#{idx + 1}</span>
              <span className="font-mono" style={{ fontSize: "0.84rem", color: aWon ? "var(--win)" : "var(--lose)", flex: 1 }}>{winName}</span>
              <span className="font-mono" style={{ fontSize: "0.80rem", color: "var(--text-dim)" }}>{ws.kills}K/{ws.deaths}D · KD {wKD}</span>
              <span className="font-mono" style={{ fontSize: "0.76rem", color: "var(--accent)" }}>DEF</span>
              <span className="font-mono" style={{ fontSize: "0.80rem", color: "var(--text-dim)" }}>{ls.kills}K/{ls.deaths}D · KD {lKD}</span>
              <span className="font-mono" style={{ fontSize: "0.84rem", color: aWon ? "var(--lose)" : "var(--win)", flex: 1, textAlign: "right" }}>{loseName}</span>
              <span className="font-mono" style={{ fontSize: "0.74rem", color: "var(--text-dim)", minWidth: 56, textAlign: "right" }}>
                {new Date(g.timestamp).toLocaleDateString()}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
