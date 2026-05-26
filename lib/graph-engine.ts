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
  assists: number;
  headshots: number; // HS% = headshots / kills
}

export interface GraphState {
  players: Record<string, Player>;
  games: Game[];
}

export interface StatVector {
  kd: number;
  kda: number;       // (K + A/2) / D
  hs_pct: number;    // headshots / kills, 0–1
  win_rate: number;  // wins / games played, 0–1
  kpr: number;       // kills per round
  games_played: number;
}

export interface PairwisePrediction {
  p_a_wins: number;
  confidence: number;
  direct_games: number;
  paths_used: number;
  evidence_mass: number;
  path_dist: Record<number, number>; // path_length -> count
  top_paths: Array<{ nodes: string[]; implied_p: number; forward: boolean }>;
}

export interface RankEntry {
  player_id: string;
  display_name: string;
  tournament_wins: number;
  tournament_win_pct: number;
  stat_vec: StatVector;
  matchup_table: Record<string, number>; // opponent_id → P(this player wins)
}

// ─── Parameters ───────────────────────────────────────────────────────────────

const TAU_DAYS = 90;           // recency decay half-life
const SIM_ROUNDS = 1000;       // round robin rounds per pairing
const L_MAX = 5;               // max path length for graph traversal
const STAT_PRIOR_WEIGHT = 0.3; // stat-based prior weight when graph evidence is thin
const DIRECT_MULTIPLIER = 2;   // direct head-to-head games outweigh indirect chains

// ─── Graph Primitives ─────────────────────────────────────────────────────────

// adj[winner][loser] = array of decayed weights for each direct game
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
// Finds all simple paths A → ... → B up to L_MAX hops.
// Each path is weighted by 1/length and contributes an implied win probability
// derived from per-hop edge win rates.

interface PathContrib {
  implied_p: number;
  path_weight: number; // 1 / path_length
  evidence: number;    // product of edge masses along the path
  nodes: string[];     // player IDs along the path, source → dest
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
        if (depth > 0) {  // exclude direct edge; handled by directP
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
// API hook: swap this body with a fetch to r6data.com per player for ranked stats.

export function computeStatVector(playerId: string, games: Game[]): StatVector {
  const relevant = games.filter(g => g.winner_id === playerId || g.loser_id === playerId);
  if (relevant.length === 0) {
    return { kd: 1, kda: 1, hs_pct: 0.3, win_rate: 0.5, kpr: 1, games_played: 0 };
  }

  let kills = 0, deaths = 0, assists = 0, hs = 0, wins = 0;
  for (const g of relevant) {
    const won = g.winner_id === playerId;
    const s = won ? g.winner_stats : g.loser_stats;
    kills += s.kills;
    deaths += s.deaths;
    assists += s.assists;
    hs += s.headshots;
    if (won) wins++;
  }

  return {
    kd:        deaths > 0 ? kills / deaths : kills,
    kda:       deaths > 0 ? (kills + assists * 0.5) / deaths : kills,
    hs_pct:    kills > 0  ? hs / kills : 0,
    win_rate:  wins / relevant.length,
    kpr:       kills / relevant.length,
    games_played: relevant.length,
  };
}

// ─── Stat-Based Prior ─────────────────────────────────────────────────────────
// P(A beats B) anchored on stat differentials. Used when graph evidence is thin.
// Weights reflect R6 1v1 relevance; logit scaled so a 1-unit KD gap ≈ 60% win probability.

function statPrior(a: StatVector, b: StatVector): number {
  const logit =
    (a.kd       - b.kd)       * 0.45 +  // kills/deaths ratio — primary skill signal
    (a.hs_pct   - b.hs_pct)   * 0.22 +  // aim consistency
    (a.win_rate - b.win_rate)  * 0.18 +  // direct outcome signal
    (a.kda      - b.kda)      * 0.10 +  // assists rarely occur in 1v1
    (a.kpr      - b.kpr)      * 0.05;   // kills per round, secondary
  return sigmoid(logit * 2.5);
}

// ─── Prediction Core ─────────────────────────────────────────────────────────
// Accepts pre-built structures so the caller can share them across many predictions.

function predictInternal(
  adj: EdgeMap,
  statA: StatVector,
  statB: StatVector,
  aId: string,
  bId: string,
): PairwisePrediction {
  const direct = directP(adj, aId, bId);
  const directGames = (adj[aId]?.[bId]?.length ?? 0) + (adj[bId]?.[aId]?.length ?? 0);
  const forwardPaths = findPaths(adj, aId, bId); // a→...→b: evidence a wins
  const reversePaths = findPaths(adj, bId, aId); // b→...→a: evidence b wins
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
    weightedP += (1 - path.implied_p) * w; // b winning this chain = a loses
  }

  const p_a_wins = Math.max(0.02, Math.min(0.98, weightedP / totalWeight));
  const graphMass = totalWeight - STAT_PRIOR_WEIGHT;
  const path_dist: Record<number, number> = {};
  if (direct) path_dist[1] = 1;
  for (const path of forwardPaths) {
    const len = Math.round(1 / path.path_weight);
    path_dist[len] = (path_dist[len] ?? 0) + 1;
  }
  for (const path of reversePaths) {
    const len = Math.round(1 / path.path_weight);
    path_dist[len] = (path_dist[len] ?? 0) + 1;
  }

  const top_paths = [
    ...forwardPaths.map(p => ({ nodes: p.nodes, implied_p: p.implied_p, forward: true,  score: p.evidence * p.path_weight })),
    ...reversePaths.map(p => ({ nodes: p.nodes, implied_p: 1 - p.implied_p, forward: false, score: p.evidence * p.path_weight })),
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

export function predictPairwise(
  state: GraphState,
  aId: string,
  bId: string,
): PairwisePrediction {
  const adj = buildEdgeMap(state.games);
  return predictInternal(
    adj,
    computeStatVector(aId, state.games),
    computeStatVector(bId, state.games),
    aId,
    bId,
  );
}

export function simulateRoundRobin(state: GraphState): RankEntry[] {
  const players = Object.values(state.players);
  if (players.length === 0) return [];

  // Build shared structures once instead of rebuilding per pair
  const adj = buildEdgeMap(state.games);
  const statVecs: Record<string, StatVector> = {};
  for (const p of players) statVecs[p.id] = computeStatVector(p.id, state.games);

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

  const champPoints: Record<string, number> = {};
  for (const p of players) champPoints[p.id] = 0;

  for (let t = 0; t < SIM_ROUNDS; t++) {
    const tourneyWins: Record<string, number> = {};
    for (const p of players) tourneyWins[p.id] = 0;

    for (let i = 0; i < players.length; i++) {
      for (let j = i + 1; j < players.length; j++) {
        const a = players[i], b = players[j];
        if (Math.random() < pMatrix[a.id][b.id]) tourneyWins[a.id]++;
        else tourneyWins[b.id]++;
      }
    }

    const maxWins = Math.max(...Object.values(tourneyWins));
    const tied = players.filter(p => tourneyWins[p.id] === maxWins);
    const share = 1 / tied.length;
    for (const w of tied) champPoints[w.id] += share;
  }

  return players
    .map(p => ({
      player_id: p.id,
      display_name: p.display_name,
      tournament_wins: champPoints[p.id],
      tournament_win_pct: champPoints[p.id] / SIM_ROUNDS,
      stat_vec: statVecs[p.id],
      matchup_table: matchupTable[p.id],
    }))
    .sort((a, b) => b.tournament_wins - a.tournament_wins);
}

export function computeElo(state: GraphState): Record<string, number> {
  const K = 32;
  const elo: Record<string, number> = {};
  for (const id of Object.keys(state.players)) elo[id] = 1000;
  for (const g of [...state.games].sort((a, b) => a.timestamp - b.timestamp)) {
    const ra = elo[g.winner_id] ?? 1000;
    const rb = elo[g.loser_id] ?? 1000;
    const ea = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    elo[g.winner_id] = Math.round(ra + K * (1 - ea));
    elo[g.loser_id]  = Math.round(rb + K * (0 - (1 - ea)));
  }
  return elo;
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
  for (const a of players) {
    for (const b of players) {
      if (a.id === b.id) continue;
      if (directP(adj, a.id, b.id)) dist[1] = (dist[1] ?? 0) + 1;
      for (const path of findPaths(adj, a.id, b.id)) {
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
