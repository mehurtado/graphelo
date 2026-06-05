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
  score_winner?: number;
  score_loser?: number;
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

export interface DominancePath {
  path: string[];              // player IDs from source to target (inclusive, ≥3 nodes = ≥1 intermediary)
  weight: number;              // average Katz value per hop
}

export interface PairwisePrediction {
  p_a_wins: number;
  confidence: number;
  direct_games: number;
  evidence_mass: number;       // katz_ab + katz_ba
  katz_ab: number;             // K[A][B]
  katz_ba: number;             // K[B][A]
  dominance_paths: DominancePath[];
  no_real_paths: boolean;      // true when no direct games AND no observed transitive paths
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
  champ_counts: Record<string, number>;
}

// ─── Extended Result Types ─────────────────────────────────────────────────────

export interface BTResult {
  ratings: Record<string, number>;       // ELO-scale BT ratings (mean=1000)
  beta: Record<string, number>;          // raw β parameters
  se: Record<string, number>;            // analytic standard errors
  ids: string[];
}

export interface CycleInfo {
  players: [string, string, string];     // [A, B, C]: A→B, B→C, C→A
  strength: number;                      // geometric mean of edge net-win weights
}

export interface CycleAnalysis {
  cycles: CycleInfo[];
  transitivity_rate: number;             // fraction of closed triples that are NOT cycles (0=all cycles, 1=fully transitive)
  total_triples: number;
}

export interface CeilingEstimate {
  ceiling: number;
  lambda: number;
  current_elo: number;
  gap_to_ceiling: number;
  plateaued: boolean;                    // current ELO within 1σ of ceiling
  se: number;
}

export interface NemesisEntry {
  opponent_id: string;
  actual_wr: number;
  expected_wr: number;
  delta: number;                         // actual - expected (negative = nemesis)
  games: number;
}

export interface NemesisProfile {
  nemesis: NemesisEntry | null;          // worst underperformance
  victim: NemesisEntry | null;           // best overperformance
  all: NemesisEntry[];
}

export interface ParityResult {
  elo_spread: number;                    // std dev of ELO ratings
  entropy: number;                       // Shannon entropy of CHAMP% distribution
  normalized_parity: number;             // 0–1, 1 = perfect parity
  max_entropy: number;
}

export interface RematchUrgency {
  prediction_shift: number;             // |p_now - p_then|
  days_since: number;
  urgency_score: number;                 // 0.7*shift + 0.3*time_decay
  direction: string;                     // who has gotten stronger
}

export interface UpsetAlertResult {
  underdog: string;
  favorite: string;
  elo_win_prob: number;
  form_win_prob: number;
  divergence: number;
  alert_level: "MODERATE" | "HIGH";
}

export interface LooCvResult {
  accuracy: number;
  brier_score: number;
  brier_skill: number;                   // improvement over 0.25 random baseline
  n: number;
  in_sample_accuracy: number;
  in_sample_n: number;
}

export interface SkillGapTrend {
  slope: number;                         // ELO spread change per game
  trend: "CONVERGING" | "DIVERGING" | "STABLE";
  current_spread: number;
  window: number;
}

// ─── Parameters ───────────────────────────────────────────────────────────────

const TAU_DAYS = 90;
const SIM_ROUNDS = 1000;
const BETA = 0.5;

// Adaptive stat prior — decays toward PRIOR_W_MIN as game data accumulates.
// PRIOR_HALF_LIFE: avg games at which prior weight halves (lower = trust data faster).
const PRIOR_W_MAX = 0.30;
const PRIOR_W_MIN = 0.05;
const PRIOR_HALF_LIFE = 4;

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

function statPriorWeight(nA: number, nB: number): number {
  const n = (nA + nB) / 2;
  return PRIOR_W_MIN + (PRIOR_W_MAX - PRIOR_W_MIN) * Math.exp(-n / PRIOR_HALF_LIFE);
}

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
  // Topology from real games only — never includes stat prior.
  // observedEdge[i][j] = true iff player i has net wins over player j.
  observedEdge: boolean[][];
}

function buildKatzMatrix(state: GraphState): KatzResult {
  const ids = Object.keys(state.players);
  const n = ids.length;
  const idx: Record<string, number> = {};
  ids.forEach((id, i) => { idx[id] = i; });

  if (n === 0) return { K: [], P: [], ids, idx, observedEdge: [] };

  const statVecs: StatVector[] = ids.map(id => computeStatVector(id, state.games));

  // W[i][j] = adaptive stat prior + decayed observed wins of i over j
  const W: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      i === j ? 0 : statPriorWeight(statVecs[i].games_played, statVecs[j].games_played) * statPrior(statVecs[i], statVecs[j])
    )
  );

  // Build observed-edge topology strictly from game history (separate from W)
  const netWins: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const g of state.games) {
    const wi = idx[g.winner_id], li = idx[g.loser_id];
    if (wi === undefined || li === undefined) continue;
    W[wi][li] += decayWeight(g.timestamp);
    netWins[wi][li]++;
  }
  const observedEdge: boolean[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => netWins[i][j] > netWins[j][i])
  );

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

  return { K, P, ids, idx, observedEdge };
}

function findObservedPaths(
  source: number,
  target: number,
  observedEdge: boolean[][],
  n: number,
  maxHops = 4,
): number[][] {
  const results: number[][] = [];
  const visited = new Array(n).fill(false);
  visited[source] = true;

  function dfs(current: number, path: number[]) {
    if (path.length > maxHops + 1) return;
    if (current === target && path.length >= 3) {
      results.push([...path]);
      return;
    }
    for (let next = 0; next < n; next++) {
      if (visited[next]) continue;
      if (!observedEdge[current][next]) continue;
      visited[next] = true;
      path.push(next);
      dfs(next, path);
      path.pop();
      visited[next] = false;
    }
  }

  dfs(source, [source]);
  return results;
}

function katzPredict(
  { K, ids, idx, observedEdge }: KatzResult,
  aId: string,
  bId: string,
  directGames: number,
): PairwisePrediction {
  const ai = idx[aId], bi = idx[bId];
  if (ai === undefined || bi === undefined) {
    return { p_a_wins: 0.5, confidence: 0, direct_games: 0, evidence_mass: 0, katz_ab: 0, katz_ba: 0, dominance_paths: [], no_real_paths: true };
  }

  const katz_ab = Math.max(0, K[ai][bi]);
  const katz_ba = Math.max(0, K[bi][ai]);
  const evidence_mass = katz_ab + katz_ba;
  const p_a_wins = Math.max(0.02, Math.min(0.98, evidence_mass > 0 ? katz_ab / evidence_mass : 0.5));
  const confidence = Math.max(0, Math.min(1, 1 - Math.exp(-evidence_mass * 0.8)));

  // DFS over real observed edges to find transitive paths (≥1 intermediary, up to 4 hops)
  const n = ids.length;
  const rawPaths = findObservedPaths(ai, bi, observedEdge, n, 4);
  const dominance_paths: DominancePath[] = rawPaths
    .map(path => {
      let w = 0;
      for (let i = 0; i < path.length - 1; i++) w += Math.max(0, K[path[i]][path[i + 1]]);
      return { path: path.map(i => ids[i]), weight: w / (path.length - 1) };
    })
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3);

  const no_real_paths = directGames === 0 && dominance_paths.length === 0;

  return {
    p_a_wins,
    confidence,
    direct_games: directGames,
    evidence_mass,
    katz_ab,
    katz_ba,
    dominance_paths,
    no_real_paths,
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
  if (players.length === 0) return { ranking: [], h2h_sim: {}, champ_counts: {} };
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

  return { ranking, h2h_sim: h2h, champ_counts: champCount };
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

// ─── Bradley-Terry (MM algorithm) ────────────────────────────────────────────

export function computeBradleyTerry(state: GraphState): BTResult | null {
  const ids = Object.keys(state.players);
  const n = ids.length;
  if (n < 2) return null;

  const idx: Record<string, number> = {};
  ids.forEach((id, i) => { idx[id] = i; });

  const now = Date.now();
  const TAU_MS = TAU_DAYS * 86_400_000;

  function gameWeight(g: Game): number {
    const age = (now - g.timestamp) / 86_400_000;
    const recency = Math.exp(-age / TAU_DAYS);
    if (g.score_winner !== undefined && g.score_loser !== undefined) {
      const margin = g.score_winner - g.score_loser;
      const rounds = g.score_winner + g.score_loser;
      const mw = rounds > 0 ? 1 + 0.1 * (margin / rounds) : 1;
      return recency * mw;
    }
    return recency;
  }

  const weights = state.games.map(gameWeight);

  const LAMBDA = 0.01;
  let beta = new Array(n).fill(0);
  const MAX_ITER = 1000;
  const TOL = 1e-6;

  for (let iter = 0; iter < MAX_ITER; iter++) {
    const betaOld = [...beta];
    for (let i = 0; i < n; i++) {
      let numerator = 0, denominator = LAMBDA;
      for (let g = 0; g < state.games.length; g++) {
        const game = state.games[g];
        const w = weights[g];
        const isWinner = game.winner_id === ids[i];
        const isLoser  = game.loser_id  === ids[i];
        if (!isWinner && !isLoser) continue;
        const j = isWinner ? idx[game.loser_id] : idx[game.winner_id];
        if (j === undefined) continue;
        const p_ij = 1 / (1 + Math.exp(beta[j] - beta[i]));
        if (isWinner) numerator += w;
        denominator += w * p_ij;
      }
      if (denominator > 0) beta[i] = Math.log(Math.max(1e-9, numerator / denominator));
    }
    const maxChange = Math.max(...beta.map((b, i) => Math.abs(b - betaOld[i])));
    if (maxChange < TOL) break;
  }

  const mean = beta.reduce((a, b) => a + b, 0) / n;
  beta = beta.map(b => b - mean);

  // Fisher information for standard errors
  const diagFisher = new Array(n).fill(0);
  for (let g = 0; g < state.games.length; g++) {
    const game = state.games[g];
    const w = weights[g];
    const i = idx[game.winner_id], j = idx[game.loser_id];
    if (i === undefined || j === undefined) continue;
    const p = 1 / (1 + Math.exp(beta[j] - beta[i]));
    const info = w * p * (1 - p);
    diagFisher[i] += info;
    diagFisher[j] += info;
  }

  const se: Record<string, number> = {};
  const ratings: Record<string, number> = {};
  const betaMap: Record<string, number> = {};
  const SCALE = 400 / Math.log(10);
  for (let i = 0; i < n; i++) {
    betaMap[ids[i]] = beta[i];
    ratings[ids[i]] = Math.round(1000 + beta[i] * SCALE);
    se[ids[i]] = diagFisher[i] > 0 ? Math.sqrt(1 / diagFisher[i]) * SCALE : 200;
  }

  return { ratings, beta: betaMap, se, ids };
}

// ─── 3-Cycle Detection ────────────────────────────────────────────────────────

export function computeCycles(state: GraphState): CycleAnalysis {
  const ids = Object.keys(state.players);
  const n = ids.length;

  // Net win counts per ordered pair (observed games only, ≥2 games to confirm edge)
  const netWins: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const totalGames: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const idx: Record<string, number> = {};
  ids.forEach((id, i) => { idx[id] = i; });

  for (const g of state.games) {
    const wi = idx[g.winner_id], li = idx[g.loser_id];
    if (wi === undefined || li === undefined) continue;
    netWins[wi][li]++;
    totalGames[wi][li]++;
    totalGames[li][wi]++;
  }

  // Edge A→B exists if A has net wins over B and total games ≥ 2
  const hasEdge = (a: number, b: number) =>
    totalGames[a][b] + totalGames[b][a] >= 2 && netWins[a][b] > netWins[b][a];

  const edgeWeight = (a: number, b: number) => {
    const total = totalGames[a][b] + totalGames[b][a];
    return total > 0 ? (netWins[a][b] - netWins[b][a]) / total : 0;
  };

  const cycles: CycleInfo[] = [];
  let transitive = 0, cyclic = 0;

  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      if (b === a || !hasEdge(a, b)) continue;
      for (let c = 0; c < n; c++) {
        if (c === a || c === b || !hasEdge(b, c)) continue;
        if (hasEdge(c, a)) {
          // A→B→C→A: cycle
          const strength = Math.pow(edgeWeight(a, b) * edgeWeight(b, c) * edgeWeight(c, a), 1 / 3);
          cycles.push({ players: [ids[a], ids[b], ids[c]], strength });
          cyclic++;
        } else if (hasEdge(a, c)) {
          transitive++;
        }
      }
    }
  }

  // Deduplicate cycles (same triple in different rotation)
  const seen = new Set<string>();
  const unique = cycles.filter(cy => {
    const key = [...cy.players].sort().join("_");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  unique.sort((a, b) => b.strength - a.strength);

  const total_triples = transitive + cyclic;
  const transitivity_rate = total_triples > 0 ? transitive / total_triples : 1;

  return { cycles: unique, transitivity_rate, total_triples };
}

// ─── Ceiling Estimate ─────────────────────────────────────────────────────────
// Fits ELO(n) = ceiling - (ceiling - 1000) * exp(-n/λ) to personal-best ELO sequence.

export function computeCeilingEstimate(
  eloValues: Array<{ timestamp: number; elo: number }>,
): CeilingEstimate | null {
  if (eloValues.length < 8) return null;

  // Build personal-best (running max) sequence
  const pbSequence: number[] = [];
  let pb = -Infinity;
  for (const { elo } of eloValues) {
    if (elo > pb) pb = elo;
    pbSequence.push(pb);
  }

  const currentElo = eloValues[eloValues.length - 1].elo;
  const maxElo = Math.max(...pbSequence);

  // Gradient descent on (ceiling, lambda)
  let ceiling = maxElo + 50;
  let lambda = eloValues.length / 2;
  const lr = 0.01;

  for (let iter = 0; iter < 2000; iter++) {
    let dCeiling = 0, dLambda = 0;
    for (let i = 0; i < pbSequence.length; i++) {
      const n = i + 1;
      const expTerm = Math.exp(-n / lambda);
      const pred = ceiling - (ceiling - 1000) * expTerm;
      const err = pred - pbSequence[i];
      dCeiling += err * (1 - expTerm);
      dLambda  += err * (ceiling - 1000) * expTerm * (-n / (lambda * lambda));
    }
    ceiling -= lr * dCeiling / pbSequence.length;
    lambda  -= lr * dLambda  / pbSequence.length;
    lambda = Math.max(0.5, lambda);
    ceiling = Math.max(currentElo, ceiling);
  }

  // Rough SE via residual std dev
  let ssRes = 0;
  for (let i = 0; i < pbSequence.length; i++) {
    const pred = ceiling - (ceiling - 1000) * Math.exp(-(i + 1) / lambda);
    ssRes += (pbSequence[i] - pred) ** 2;
  }
  const se = Math.sqrt(ssRes / pbSequence.length);

  return {
    ceiling: Math.round(ceiling),
    lambda: Math.round(lambda * 10) / 10,
    current_elo: currentElo,
    gap_to_ceiling: Math.round(ceiling - currentElo),
    plateaued: (ceiling - currentElo) < se,
    se: Math.round(se),
  };
}

// ─── Nemesis / Victim ─────────────────────────────────────────────────────────

export function computeNemesisProfile(state: GraphState, playerId: string): NemesisProfile {
  const elo = computeElo(state);
  const opponents = Object.keys(state.players).filter(id => id !== playerId);
  const entries: NemesisEntry[] = [];

  for (const oppId of opponents) {
    const h2hGames = state.games.filter(g =>
      (g.winner_id === playerId && g.loser_id === oppId) ||
      (g.winner_id === oppId   && g.loser_id === playerId)
    );
    if (h2hGames.length < 3) continue;

    const wins = h2hGames.filter(g => g.winner_id === playerId).length;
    const actual_wr = wins / h2hGames.length;
    const rA = elo[playerId] ?? 1000, rB = elo[oppId] ?? 1000;
    const expected_wr = 1 / (1 + Math.pow(10, (rB - rA) / 400));

    entries.push({
      opponent_id: oppId,
      actual_wr,
      expected_wr,
      delta: actual_wr - expected_wr,
      games: h2hGames.length,
    });
  }

  entries.sort((a, b) => a.delta - b.delta);
  return {
    nemesis: entries[0]?.delta < -0.1 ? entries[0] : null,
    victim:  entries[entries.length - 1]?.delta > 0.1 ? entries[entries.length - 1] : null,
    all: entries,
  };
}

// ─── Parity Index ─────────────────────────────────────────────────────────────

export function computeParityIndex(
  state: GraphState,
  champCounts: Record<string, number>,
  simRounds: number,
): ParityResult {
  const ids = Object.keys(state.players);
  const n = ids.length;
  if (n < 2) return { elo_spread: 0, entropy: 0, normalized_parity: 0, max_entropy: Math.log2(n) };

  const elo = computeElo(state);
  const eloVals = ids.map(id => elo[id] ?? 1000);
  const mean = eloVals.reduce((a, b) => a + b, 0) / n;
  const elo_spread = Math.sqrt(eloVals.reduce((s, e) => s + (e - mean) ** 2, 0) / n);

  const probs = ids.map(id => (champCounts[id] ?? 0) / simRounds);
  const entropy = -probs.reduce((s, p) => p > 0 ? s + p * Math.log2(p) : s, 0);
  const max_entropy = Math.log2(n);
  const normalized_parity = max_entropy > 0 ? entropy / max_entropy : 0;

  return { elo_spread, entropy, normalized_parity, max_entropy };
}

// ─── Form Rating ──────────────────────────────────────────────────────────────
// Short-window ELO: recompute ELO using only each player's last N games.

export function computeFormRating(state: GraphState, window = 5): Record<string, number> {
  const ids = Object.keys(state.players);
  const result: Record<string, number> = {};

  for (const id of ids) {
    const myGames = [...state.games]
      .filter(g => g.winner_id === id || g.loser_id === id)
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(-window);

    if (myGames.length === 0) { result[id] = 1000; continue; }

    // Build mini-state with only the players in those games
    const miniPlayers: Record<string, { id: string; display_name: string; created_at: number }> = {};
    for (const g of myGames) {
      if (state.players[g.winner_id]) miniPlayers[g.winner_id] = state.players[g.winner_id];
      if (state.players[g.loser_id])  miniPlayers[g.loser_id]  = state.players[g.loser_id];
    }
    const miniElo = computeElo({ players: miniPlayers, games: myGames });
    result[id] = miniElo[id] ?? 1000;
  }
  return result;
}

// ─── LOO / Temporal Brier Score ───────────────────────────────────────────────
// For each game, predict using only games before it (temporal holdout).

export function computeLooCvBrier(state: GraphState): LooCvResult | null {
  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  if (sorted.length < 4) return null;

  let brierSum = 0, correct = 0, n = 0;
  for (let i = 1; i < sorted.length; i++) {
    const g = sorted[i];
    if (!state.players[g.winner_id] || !state.players[g.loser_id]) continue;
    const hist: GraphState = { players: state.players, games: sorted.slice(0, i) };
    const { p_a_wins } = predictPairwise(hist, g.winner_id, g.loser_id);
    brierSum += (p_a_wins - 1) ** 2;
    if (p_a_wins > 0.5) correct++;
    n++;
  }
  if (n === 0) return null;

  const brier_score = brierSum / n;
  const brier_baseline = 0.25;
  const brier_skill = 1 - brier_score / brier_baseline;

  // In-sample
  const acc = computePredictionAccuracy(state);

  return {
    accuracy: correct / n,
    brier_score,
    brier_skill,
    n,
    in_sample_accuracy: acc.total > 0 ? acc.correct / acc.total : 0,
    in_sample_n: acc.total,
  };
}

// ─── Skill Gap Trend ──────────────────────────────────────────────────────────

export function computeSkillGapTrend(state: GraphState, windowGames = 10): SkillGapTrend | null {
  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  const ids = Object.keys(state.players);
  if (sorted.length < windowGames + 1 || ids.length < 2) return null;

  const spreads: number[] = [];
  for (let i = Math.max(2, sorted.length - windowGames); i <= sorted.length; i++) {
    const partial = computeElo({ players: state.players, games: sorted.slice(0, i) });
    const vals = ids.map(id => partial[id] ?? 1000);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    spreads.push(Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length));
  }

  const m = spreads.length;
  const xMean = (m - 1) / 2;
  const yMean = spreads.reduce((a, b) => a + b, 0) / m;
  const num = spreads.reduce((s, y, i) => s + (i - xMean) * (y - yMean), 0);
  const den = spreads.reduce((s, _, i) => s + (i - xMean) ** 2, 0);
  const slope = den > 0 ? num / den : 0;

  return {
    slope: Math.round(slope * 10) / 10,
    trend: slope > 1 ? "DIVERGING" : slope < -1 ? "CONVERGING" : "STABLE",
    current_spread: Math.round(spreads[spreads.length - 1]),
    window: m,
  };
}

// ─── Rematch Urgency ──────────────────────────────────────────────────────────

export function computeRematchUrgency(
  state: GraphState,
  aId: string,
  bId: string,
): RematchUrgency | null {
  const h2hGames = state.games
    .filter(g => (g.winner_id === aId && g.loser_id === bId) || (g.winner_id === bId && g.loser_id === aId))
    .sort((a, b) => a.timestamp - b.timestamp);
  if (h2hGames.length === 0) return null;

  const lastGame = h2hGames[h2hGames.length - 1];
  const eloAtLast = computeElo({ players: state.players, games: state.games.filter(g => g.timestamp <= lastGame.timestamp) });
  const rA = eloAtLast[aId] ?? 1000, rB = eloAtLast[bId] ?? 1000;
  const predictedThen = 1 / (1 + Math.pow(10, (rB - rA) / 400));

  const { p_a_wins: predictedNow } = predictPairwise(state, aId, bId);
  const daysSince = (Date.now() - lastGame.timestamp) / 86_400_000;
  const prediction_shift = Math.abs(predictedNow - predictedThen);
  const time_decay = 1 - Math.exp(-daysSince / 30);
  const urgency_score = 0.7 * prediction_shift + 0.3 * time_decay;

  const aName = state.players[aId]?.display_name ?? aId;
  const bName = state.players[bId]?.display_name ?? bId;
  const direction = predictedNow > predictedThen
    ? `${aName} stronger now`
    : `${bName} stronger now`;

  return {
    prediction_shift: Math.round(prediction_shift * 100) / 100,
    days_since: Math.round(daysSince),
    urgency_score: Math.round(urgency_score * 100) / 100,
    direction,
  };
}

// ─── Upset Alert ──────────────────────────────────────────────────────────────

export function computeUpsetAlert(
  aId: string,
  bId: string,
  katzWinProb: number,
  formRating: Record<string, number>,
): UpsetAlertResult | null {
  const fA = formRating[aId] ?? 1000, fB = formRating[bId] ?? 1000;
  const form_win_prob = 1 / (1 + Math.pow(10, (fB - fA) / 400));
  const divergence = Math.abs(form_win_prob - katzWinProb);
  if (divergence < 0.15) return null;

  const underdog = form_win_prob > katzWinProb ? aId : bId;
  const favorite = underdog === aId ? bId : aId;

  return {
    underdog,
    favorite,
    elo_win_prob: katzWinProb,
    form_win_prob,
    divergence: Math.round(divergence * 100) / 100,
    alert_level: divergence > 0.25 ? "HIGH" : "MODERATE",
  };
}
