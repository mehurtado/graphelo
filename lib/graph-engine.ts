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
  evidence_mass: number;       // katz_ab + katz_ba
  katz_ab: number;             // K[A][B]
  katz_ba: number;             // K[B][A]
  top_contributors: Array<{ playerId: string; mass: number }>;
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
const BETA = 0.5;
const STAT_PRIOR_WEIGHT = 0.3;

// ─── Utilities ────────────────────────────────────────────────────────────────

function decayWeight(timestamp: number): number {
  return Math.exp(-(Date.now() - timestamp) / (TAU_DAYS * 86_400_000));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
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

// ─── Matrix Inversion (Gauss-Jordan, partial pivoting) ────────────────────────

function invertMatrix(A: number[][]): number[][] {
  const n = A.length;
  const M = A.map((row, i) =>
    [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]
  );
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) {
      // Singular — return identity
      return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
      );
    }
    for (let j = col; j < 2 * n; j++) M[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col];
      for (let j = col; j < 2 * n; j++) M[row][j] -= factor * M[col][j];
    }
  }
  return M.map(row => row.slice(n));
}

// ─── Katz Graph Diffusion ─────────────────────────────────────────────────────
// K = (I − βP)^{−1} − I, where P is row-stochastic win matrix.
// K[i][j] = sum of all walks from i to j, each weighted by β^(walk length).

interface KatzResult {
  K: number[][];
  P: number[][];
  ids: string[];
  idx: Record<string, number>;
}

function buildKatzMatrix(state: GraphState): KatzResult {
  const ids = Object.keys(state.players);
  const n = ids.length;
  const idx: Record<string, number> = {};
  ids.forEach((id, i) => { idx[id] = i; });

  if (n === 0) return { K: [], P: [], ids, idx };

  const statVecs: StatVector[] = ids.map(id => computeStatVector(id, state.games));

  // W[i][j] = stat prior + decayed observed wins of i over j
  const W: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      i === j ? 0 : STAT_PRIOR_WEIGHT * statPrior(statVecs[i], statVecs[j])
    )
  );
  for (const g of state.games) {
    const wi = idx[g.winner_id], li = idx[g.loser_id];
    if (wi === undefined || li === undefined) continue;
    W[wi][li] += decayWeight(g.timestamp);
  }

  // Row-normalize → stochastic P
  const P: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const rowSum = W[i].reduce((s, v) => s + v, 0);
    if (rowSum > 0) for (let j = 0; j < n; j++) P[i][j] = W[i][j] / rowSum;
  }

  // Build (I − β·P) and invert
  const IminusBP: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - BETA * P[i][j])
  );
  const inv = invertMatrix(IminusBP);

  // K = inv − I
  const K: number[][] = inv.map((row, i) =>
    row.map((v, j) => v - (i === j ? 1 : 0))
  );

  return { K, P, ids, idx };
}

function katzPredict(
  { K, P, ids, idx }: KatzResult,
  aId: string,
  bId: string,
  directGames: number,
): PairwisePrediction {
  const ai = idx[aId], bi = idx[bId];
  if (ai === undefined || bi === undefined) {
    return { p_a_wins: 0.5, confidence: 0, direct_games: 0, evidence_mass: 0, katz_ab: 0, katz_ba: 0, top_contributors: [] };
  }

  const katz_ab = Math.max(0, K[ai][bi]);
  const katz_ba = Math.max(0, K[bi][ai]);
  const evidence_mass = katz_ab + katz_ba;
  const p_a_wins = Math.max(0.02, Math.min(0.98, evidence_mass > 0 ? katz_ab / evidence_mass : 0.5));
  const confidence = Math.max(0, Math.min(1, 1 - Math.exp(-evidence_mass * 0.8)));

  // Top-2 contributors via 2-hop K mass: β² × P[A→m] × P[m→B]
  const n = ids.length;
  const contributors: Array<{ playerId: string; mass: number }> = [];
  for (let m = 0; m < n; m++) {
    if (m === ai || m === bi) continue;
    const mass = BETA * BETA * P[ai][m] * P[m][bi];
    if (mass > 0) contributors.push({ playerId: ids[m], mass });
  }
  contributors.sort((a, b) => b.mass - a.mass);

  return {
    p_a_wins,
    confidence,
    direct_games: directGames,
    evidence_mass,
    katz_ab,
    katz_ba,
    top_contributors: contributors.slice(0, 2),
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function predictPairwise(state: GraphState, aId: string, bId: string): PairwisePrediction {
  const katz = buildKatzMatrix(state);
  const directGames = state.games.filter(g =>
    (g.winner_id === aId && g.loser_id === bId) ||
    (g.winner_id === bId && g.loser_id === aId)
  ).length;
  return katzPredict(katz, aId, bId, directGames);
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
  const lossWeight = new Array(n).fill(0);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++)
      lossWeight[i] += W[j][i];

  // Power iteration
  // Transition: from i, go to j with prob W[j][i] / lossWeight[i]
  let pr = new Array(n).fill(1 / n);
  for (let iter = 0; iter < 300; iter++) {
    const next = new Array(n).fill((1 - d) / n);
    for (let i = 0; i < n; i++) {
      const lw = lossWeight[i];
      if (lw === 0) {
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

  const recent = history.slice(-5);
  const m = recent.length;
  const ys = recent.map(p => p.elo);
  const xMean = (m - 1) / 2;
  const yMean = ys.reduce((s, y) => s + y, 0) / m;
  const num = ys.reduce((s, y, i) => s + (i - xMean) * (y - yMean), 0);
  const den = ys.reduce((s, _, i) => s + (i - xMean) ** 2, 0);
  const slope = den > 0 ? num / den : 0;

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

  // Build Katz matrix once for all pairs
  const katz = buildKatzMatrix(state);

  const statVecs: Record<string, StatVector> = {};
  for (const p of players) statVecs[p.id] = computeStatVector(p.id, state.games);

  const pMatrix: Record<string, Record<string, number>> = {};
  const matchupTable: Record<string, Record<string, number>> = {};
  for (const a of players) {
    pMatrix[a.id] = {};
    matchupTable[a.id] = {};
    for (const b of players) {
      if (a.id === b.id) continue;
      const { p_a_wins } = katzPredict(katz, a.id, b.id, 0);
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

    const sorted = [...players].sort((a, b) => wins[b.id] - wins[a.id]);
    sorted.forEach((p, rank) => { placementCounts[p.id][rank]++; });

    const maxWins = Math.max(...players.map(p => wins[p.id]));
    const leaders = players.filter(p => wins[p.id] === maxWins);
    if (leaders.length === 1) champCount[leaders[0].id]++;
  }

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
      player_id:          p.id,
      display_name:       p.display_name,
      tournament_wins:    champCount[p.id],
      tournament_win_pct: champCount[p.id] / SIM_ROUNDS,
      placement_dist:     placementCounts[p.id],
      stat_vec:           statVecs[p.id],
      matchup_table:      matchupTable[p.id],
      page_rank:          pageRank[p.id] ?? 0,
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
