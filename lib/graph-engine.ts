// ─── Types ────────────────────────────────────────────────────────────────────

export interface Player {
  id: string;
  display_name: string;
  created_at: number;
}

export interface Game {
  id: string;
  timestamp: number;
  winner_id: string;
  loser_id: string;
  winner_stats: PerGameStats;
  loser_stats: PerGameStats;
}

export interface PerGameStats {
  kills: number;
  deaths: number;
}

export interface GraphState {
  players: Record<string, Player>;
  games: Game[];
}

export interface StatVector {
  kd: number;
  win_rate: number;
  kpr: number;
  games_played: number;
}

export interface PairwisePrediction {
  p_a_wins: number;
  confidence: number;
  direct_games: number;
  paths_used: number;
  evidence_mass: number;
  path_dist: Record<number, number>;
  top_paths: Array<{ nodes: string[]; implied_p: number; forward: boolean }>;
}

export interface RankEntry {
  player_id: string;
  display_name: string;
  tournament_wins: number;    // exact integer: times as sole champion
  tournament_win_pct: number; // champ_count / SIM_ROUNDS
  placement_dist: number[];   // placement_dist[k] = # runs finishing in position k (0-indexed)
  stat_vec: StatVector;
  matchup_table: Record<string, number>;
  page_rank: number;          // reverse PageRank score (higher = more dominant)
}

// H2H simulation counts: h2h_sim[a][b] = # simulated matches a beat b across all runs
export type H2hSim = Record<string, Record<string, number>>;

export interface SimulationResult {
  ranking: RankEntry[];
  h2h_sim: H2hSim;
}

// ─── Parameters ───────────────────────────────────────────────────────────────

const TAU_DAYS = 90;
const SIM_ROUNDS = 1000;
const L_MAX = 5;
const STAT_PRIOR_WEIGHT = 0.3;
const DIRECT_MULTIPLIER = 2;

// ─── Graph Primitives ─────────────────────────────────────────────────────────

interface EdgeMap {
  [winnerId: string]: { [loserId: string]: number[] };
}

function decayWeight(timestamp: number): number {
  return Math.exp(-(Date.now() - timestamp) / (TAU_DAYS * 86_400_000));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function buildEdgeMap(games: Game[]): EdgeMap {
  const adj: EdgeMap = {};
  for (const g of games) {
    const w = decayWeight(g.timestamp);
    adj[g.winner_id] ??= {};
    adj[g.winner_id][g.loser_id] ??= [];
    adj[g.winner_id][g.loser_id].push(w);
  }
  return adj;
}

function totalMass(adj: EdgeMap, from: string, to: string): number {
  return (adj[from]?.[to] ?? []).reduce((s, w) => s + w, 0);
}

function directP(adj: EdgeMap, a: string, b: string): { p: number; mass: number } | null {
  const aWins = totalMass(adj, a, b);
  const bWins = totalMass(adj, b, a);
  const total = aWins + bWins;
  if (total === 0) return null;
  return { p: aWins / total, mass: total };
}

// ─── Path Traversal ───────────────────────────────────────────────────────────

interface PathContrib {
  implied_p: number;
  path_weight: number;
  evidence: number;
  nodes: string[];
}

function findPaths(adj: EdgeMap, src: string, dst: string): PathContrib[] {
  const results: PathContrib[] = [];

  function dfs(
    current: string,
    visited: Set<string>,
    depth: number,
    runningP: number,
    runningE: number,
    path: string[],
  ) {
    if (depth > L_MAX) return;
    for (const next of Object.keys(adj[current] ?? {})) {
      if (visited.has(next)) continue;
      const mass = totalMass(adj, current, next);
      if (mass === 0) continue;
      const pEdge = mass / (mass + totalMass(adj, next, current));
      if (next === dst) {
        if (depth > 0) {
          results.push({
            implied_p: runningP * pEdge,
            path_weight: 1 / (depth + 1),
            evidence: runningE * mass,
            nodes: [...path, current, next],
          });
        }
      } else {
        visited.add(next);
        dfs(next, visited, depth + 1, runningP * pEdge, runningE * mass, [...path, current]);
        visited.delete(next);
      }
    }
  }

  dfs(src, new Set([src]), 0, 1.0, 1.0, []);
  return results;
}

// ─── Stat Vector ──────────────────────────────────────────────────────────────

export function computeStatVector(playerId: string, games: Game[]): StatVector {
  const relevant = games.filter(g => g.winner_id === playerId || g.loser_id === playerId);
  if (relevant.length === 0) return { kd: 1, win_rate: 0.5, kpr: 1, games_played: 0 };

  let kills = 0, deaths = 0, wins = 0;
  for (const g of relevant) {
    const won = g.winner_id === playerId;
    const s = won ? g.winner_stats : g.loser_stats;
    kills += s.kills;
    deaths += s.deaths;
    if (won) wins++;
  }

  return {
    kd:           deaths > 0 ? kills / deaths : kills,
    win_rate:     wins / relevant.length,
    kpr:          kills / relevant.length,
    games_played: relevant.length,
  };
}

// ─── Stat-Based Prior ─────────────────────────────────────────────────────────

function statPrior(a: StatVector, b: StatVector): number {
  const logit =
    (a.win_rate - b.win_rate) * 1.50 +
    (a.kd       - b.kd)       * 0.35 +
    (a.kpr      - b.kpr)      * 0.05;
  return sigmoid(logit);
}

// ─── Prediction Core ──────────────────────────────────────────────────────────

function predictInternal(
  adj: EdgeMap,
  statA: StatVector,
  statB: StatVector,
  aId: string,
  bId: string,
): PairwisePrediction {
  const direct = directP(adj, aId, bId);
  const directGames = (adj[aId]?.[bId]?.length ?? 0) + (adj[bId]?.[aId]?.length ?? 0);
  const forwardPaths = findPaths(adj, aId, bId);
  const reversePaths = findPaths(adj, bId, aId);
  const prior = statPrior(statA, statB);

  let totalWeight = STAT_PRIOR_WEIGHT;
  let weightedP = prior * STAT_PRIOR_WEIGHT;

  if (direct) {
    totalWeight += direct.mass * DIRECT_MULTIPLIER;
    weightedP += direct.p * direct.mass * DIRECT_MULTIPLIER;
  }
  for (const path of forwardPaths) {
    const w = path.evidence * path.path_weight;
    totalWeight += w;
    weightedP += path.implied_p * w;
  }
  for (const path of reversePaths) {
    const w = path.evidence * path.path_weight;
    totalWeight += w;
    weightedP += (1 - path.implied_p) * w;
  }

  // Shrink probability toward 0.5 based on minimum path length.
  // Transitive inferences through long chains are less reliable than direct evidence.
  let minPathLen = direct ? 1 : Infinity;
  for (const path of [...forwardPaths, ...reversePaths]) {
    const len = Math.round(1 / path.path_weight);
    if (len < minPathLen) minPathLen = len;
  }
  if (!isFinite(minPathLen)) minPathLen = 5;
  const SHRINK: Record<number, number> = { 1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45, 5: 0.30 };
  const shrink = SHRINK[Math.min(5, minPathLen)] ?? 0.30;
  const p_raw = Math.max(0.02, Math.min(0.98, weightedP / totalWeight));
  const p_a_wins = 0.5 + (p_raw - 0.5) * shrink;
  const graphMass = totalWeight - STAT_PRIOR_WEIGHT;
  const path_dist: Record<number, number> = {};
  if (direct) path_dist[1] = 1;
  for (const path of [...forwardPaths, ...reversePaths]) {
    const len = Math.round(1 / path.path_weight);
    path_dist[len] = (path_dist[len] ?? 0) + 1;
  }

  const top_paths = [
    ...forwardPaths.map(p => ({ nodes: p.nodes, implied_p: p.implied_p,       forward: true,  score: p.evidence * p.path_weight })),
    ...reversePaths.map(p => ({ nodes: p.nodes, implied_p: 1 - p.implied_p,   forward: false, score: p.evidence * p.path_weight })),
  ].sort((a, b) => b.score - a.score).slice(0, 4).map(({ nodes, implied_p, forward }) => ({ nodes, implied_p, forward }));

  const totalPaths = (direct ? 1 : 0) + forwardPaths.length + reversePaths.length;
  const confidence = 1 - Math.exp(-0.15 * graphMass * Math.sqrt(1 + totalPaths));

  return {
    p_a_wins,
    confidence: Math.max(0, Math.min(1, confidence)),
    direct_games: directGames,
    paths_used: totalPaths,
    evidence_mass: totalWeight,
    path_dist,
    top_paths,
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function predictPairwise(state: GraphState, aId: string, bId: string): PairwisePrediction {
  const adj = buildEdgeMap(state.games);
  return predictInternal(
    adj,
    computeStatVector(aId, state.games),
    computeStatVector(bId, state.games),
    aId,
    bId,
  );
}

// ─── Reverse PageRank ─────────────────────────────────────────────────────────
// Random walker at node i follows "who beat i" edges with probability proportional
// to decayed win weight. Stationary distribution = authority of each player.
// High score = beaten by many other high-authority players = dominant.

export function computePageRank(state: GraphState, d = 0.85): Record<string, number> {
  const ids = Object.keys(state.players);
  const n = ids.length;
  if (n === 0) return {};

  const idx: Record<string, number> = {};
  ids.forEach((id, i) => { idx[id] = i; });

  // W[a][b] = total decayed weight of player a beating player b
  const W: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const g of state.games) {
    const wi = idx[g.winner_id], li = idx[g.loser_id];
    if (wi === undefined || li === undefined) continue;
    W[wi][li] += decayWeight(g.timestamp);
  }

  // lossWeight[i] = total decayed weight of all losses player i has suffered
  // = Σ_j W[j][i]
  const lossWeight = new Array(n).fill(0);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++)
      lossWeight[i] += W[j][i];

  // Power iteration
  // Transition: from i, go to j with prob W[j][i] / lossWeight[i]
  // (follow who-beat-me edges; authority accumulates at players who beat many)
  let pr = new Array(n).fill(1 / n);
  for (let iter = 0; iter < 300; iter++) {
    const next = new Array(n).fill((1 - d) / n);
    for (let i = 0; i < n; i++) {
      const lw = lossWeight[i];
      if (lw === 0) {
        // Never lost — distribute mass uniformly (dangling node)
        for (let j = 0; j < n; j++) next[j] += d * pr[i] / n;
      } else {
        for (let j = 0; j < n; j++) {
          if (W[j][i] > 0) next[j] += d * pr[i] * (W[j][i] / lw);
        }
      }
    }
    const diff = pr.reduce((s, v, i) => s + Math.abs(v - next[i]), 0);
    pr = next;
    if (diff < 1e-6) break;
  }

  const result: Record<string, number> = {};
  ids.forEach((id, i) => { result[id] = pr[i]; });
  return result;
}

// ─── Kendall Tau Agreement ────────────────────────────────────────────────────
// Returns 0–1: 1 = perfect agreement, 0 = perfect disagreement.

export function kendallTauAgreement(orderA: string[], orderB: string[]): number {
  const n = orderA.length;
  if (n <= 1) return 1;
  const posB: Record<string, number> = {};
  orderB.forEach((id, i) => { posB[id] = i; });
  let concordant = 0, total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const bi = posB[orderA[i]] ?? 0, bj = posB[orderA[j]] ?? 0;
      if (bi !== bj) { total++; if (bi < bj) concordant++; }
    }
  }
  return total === 0 ? 1 : concordant / total;
}

// ─── Reliability Score ────────────────────────────────────────────────────────
// 1 - exp(-n_effective / k), k=5. Games older than 2τ contribute 0.5 weight.

export function computeReliability(playerId: string, games: Game[]): number {
  const TWO_TAU_MS = 2 * TAU_DAYS * 86_400_000;
  const relevant = games.filter(g => g.winner_id === playerId || g.loser_id === playerId);
  const nEff = relevant.reduce((s, g) => s + (Date.now() - g.timestamp > TWO_TAU_MS ? 0.5 : 1.0), 0);
  return 1 - Math.exp(-nEff / 5);
}

// ─── ELO Velocity ─────────────────────────────────────────────────────────────
// slope: linear regression on last 5 ELO snapshots (ELO units per game)
// ewma:  exponentially weighted moving average of per-game ELO deltas (λ=0.3)

export function computeEloVelocity(
  history: Array<{ timestamp: number; elo: number }>,
): { slope: number; ewma: number } | null {
  if (history.length < 2) return null;

  // Linear regression on last 5 points
  const recent = history.slice(-5);
  const m = recent.length;
  const ys = recent.map(p => p.elo);
  const xMean = (m - 1) / 2;
  const yMean = ys.reduce((s, y) => s + y, 0) / m;
  const num = ys.reduce((s, y, i) => s + (i - xMean) * (y - yMean), 0);
  const den = ys.reduce((s, _, i) => s + (i - xMean) ** 2, 0);
  const slope = den > 0 ? num / den : 0;

  // EWMA of per-game deltas
  const lambda = 0.3;
  let ewma = history[1].elo - history[0].elo;
  for (let i = 2; i < history.length; i++) {
    ewma = lambda * (history[i].elo - history[i - 1].elo) + (1 - lambda) * ewma;
  }

  return { slope, ewma };
}

// ─── Monte Carlo Simulation ───────────────────────────────────────────────────

export function simulateRoundRobin(state: GraphState): SimulationResult {
  const players = Object.values(state.players);
  if (players.length === 0) return { ranking: [], h2h_sim: {} };
  const n = players.length;

  const adj = buildEdgeMap(state.games);
  const statVecs: Record<string, StatVector> = {};
  for (const p of players) statVecs[p.id] = computeStatVector(p.id, state.games);

  // Build probability and matchup tables once
  const pMatrix: Record<string, Record<string, number>> = {};
  const matchupTable: Record<string, Record<string, number>> = {};
  for (const a of players) {
    pMatrix[a.id] = {};
    matchupTable[a.id] = {};
    for (const b of players) {
      if (a.id === b.id) continue;
      const { p_a_wins } = predictInternal(adj, statVecs[a.id], statVecs[b.id], a.id, b.id);
      pMatrix[a.id][b.id] = p_a_wins;
      matchupTable[a.id][b.id] = p_a_wins;
    }
  }

  // Accumulators
  const placementCounts: Record<string, number[]> = {};
  const champCount: Record<string, number> = {};
  const h2h: H2hSim = {};
  for (const p of players) {
    placementCounts[p.id] = new Array(n).fill(0);
    champCount[p.id] = 0;
    h2h[p.id] = {};
    for (const q of players) if (q.id !== p.id) h2h[p.id][q.id] = 0;
  }

  for (let t = 0; t < SIM_ROUNDS; t++) {
    const wins: Record<string, number> = {};
    for (const p of players) wins[p.id] = 0;

    for (let i = 0; i < players.length; i++) {
      for (let j = i + 1; j < players.length; j++) {
        const a = players[i], b = players[j];
        if (Math.random() < pMatrix[a.id][b.id]) {
          wins[a.id]++;
          h2h[a.id][b.id]++;
        } else {
          wins[b.id]++;
          h2h[b.id][a.id]++;
        }
      }
    }

    // Record placements (sort descending by wins; ties broken arbitrarily)
    const sorted = [...players].sort((a, b) => wins[b.id] - wins[a.id]);
    sorted.forEach((p, rank) => { placementCounts[p.id][rank]++; });

    // Sole champion only
    const maxWins = Math.max(...players.map(p => wins[p.id]));
    const leaders = players.filter(p => wins[p.id] === maxWins);
    if (leaders.length === 1) champCount[leaders[0].id]++;
  }

  // Median placement for sort key
  const medianRank = (dist: number[]) => {
    let cum = 0;
    for (let i = 0; i < dist.length; i++) {
      cum += dist[i];
      if (cum * 2 >= SIM_ROUNDS) return i;
    }
    return dist.length - 1;
  };

  const pageRank = computePageRank(state);

  const ranking: RankEntry[] = players
    .map(p => ({
      player_id:        p.id,
      display_name:     p.display_name,
      tournament_wins:  champCount[p.id],
      tournament_win_pct: champCount[p.id] / SIM_ROUNDS,
      placement_dist:   placementCounts[p.id],
      stat_vec:         statVecs[p.id],
      matchup_table:    matchupTable[p.id],
      page_rank:        pageRank[p.id] ?? 0,
    }))
    .sort((a, b) =>
      b.tournament_win_pct - a.tournament_win_pct ||
      medianRank(a.placement_dist) - medianRank(b.placement_dist)
    );

  return { ranking, h2h_sim: h2h };
}

// ─── ELO ─────────────────────────────────────────────────────────────────────

export function computeElo(state: GraphState): Record<string, number> {
  const K = 32;
  const elo: Record<string, number> = {};
  for (const id of Object.keys(state.players)) elo[id] = 1000;
  for (const g of [...state.games].sort((a, b) => a.timestamp - b.timestamp)) {
    const ra = elo[g.winner_id] ?? 1000, rb = elo[g.loser_id] ?? 1000;
    const ea = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    elo[g.winner_id] = Math.round(ra + K * (1 - ea));
    elo[g.loser_id]  = Math.round(rb + K * (0 - (1 - ea)));
  }
  return elo;
}

export function computeEloHistory(state: GraphState): Record<string, Array<{ timestamp: number; elo: number }>> {
  const K = 32;
  const current: Record<string, number> = {};
  const history: Record<string, Array<{ timestamp: number; elo: number }>> = {};
  for (const id of Object.keys(state.players)) { current[id] = 1000; history[id] = []; }
  for (const g of [...state.games].sort((a, b) => a.timestamp - b.timestamp)) {
    const ra = current[g.winner_id] ?? 1000, rb = current[g.loser_id] ?? 1000;
    const ea = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    current[g.winner_id] = Math.round(ra + K * (1 - ea));
    current[g.loser_id]  = Math.round(rb + K * (0 - (1 - ea)));
    if (history[g.winner_id]) history[g.winner_id].push({ timestamp: g.timestamp, elo: current[g.winner_id] });
    if (history[g.loser_id])  history[g.loser_id].push({  timestamp: g.timestamp, elo: current[g.loser_id] });
  }
  return history;
}

export function computePredictionAccuracy(state: GraphState): { correct: number; total: number } {
  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  let correct = 0, total = 0;
  for (let i = 1; i < sorted.length; i++) {
    const g = sorted[i];
    if (!state.players[g.winner_id] || !state.players[g.loser_id]) continue;
    const { p_a_wins } = predictPairwise({ players: state.players, games: sorted.slice(0, i) }, g.winner_id, g.loser_id);
    if (p_a_wins > 0.5) correct++;
    total++;
  }
  return { correct, total };
}

export function computeGlobalPathDist(state: GraphState): Record<number, number> {
  const players = Object.values(state.players);
  if (players.length < 2) return {};
  const adj = buildEdgeMap(state.games);
  const dist: Record<number, number> = {};
  for (let i = 0; i < players.length; i++) {
    for (let j = i + 1; j < players.length; j++) {
      const a = players[i], b = players[j];
      if (directP(adj, a.id, b.id)) dist[1] = (dist[1] ?? 0) + 1;
      for (const path of [...findPaths(adj, a.id, b.id), ...findPaths(adj, b.id, a.id)]) {
        const len = Math.round(1 / path.path_weight);
        dist[len] = (dist[len] ?? 0) + 1;
      }
    }
  }
  return dist;
}

export function headToHead(state: GraphState, aId: string, bId: string) {
  const games = state.games.filter(
    g => (g.winner_id === aId && g.loser_id === bId) ||
         (g.winner_id === bId && g.loser_id === aId)
  );
  const aWins = games.filter(g => g.winner_id === aId).length;
  return {
    games_played: games.length,
    a_wins: aWins,
    b_wins: games.length - aWins,
    prediction: predictPairwise(state, aId, bId),
    recent_games: games.slice(-10).reverse(),
  };
}

export function computeMetaStability(state: GraphState): number | null {
  const ids = Object.keys(state.players);
  if (ids.length < 2 || state.games.length < 6) return null;

  const mid = Math.floor(state.games.length / 2);
  const eloEarly = computeElo({ players: state.players, games: state.games.slice(0, mid) });
  const eloLate  = computeElo({ players: state.players, games: state.games.slice(mid) });

  const rankOf = (eloMap: Record<string, number>) => {
    const sorted = [...ids].sort((a, b) => (eloMap[b] ?? 1000) - (eloMap[a] ?? 1000));
    return sorted.reduce<Record<string, number>>((acc, id, i) => { acc[id] = i; return acc; }, {});
  };

  const rEarly = rankOf(eloEarly), rLate = rankOf(eloLate);
  let dSq = 0;
  for (const id of ids) dSq += ((rEarly[id] ?? 0) - (rLate[id] ?? 0)) ** 2;
  const n = ids.length;
  const rho = 1 - (6 * dSq) / (n * (n * n - 1));
  return Math.max(0, Math.min(1, (rho + 1) / 2));
}
