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
  avg_pairwise_win_prob: number; // W_i: mean simulated head-to-head win rate vs. every other player (0-1)
  avg_placement: number;         // P_i: mean tournament finish, 1-indexed (1 = always wins, N = always last)
  champ_pct_norm: number;        // C_i normalized against the theoretical fair share (1/N), capped at 1
  sim_score: number;              // composite ranking score, see SIMULATION_SCORE_ALPHA
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
export const ELO_START = 800;
const K_STABLE = 32;
const K_PROVISIONAL = 64;
const PROVISIONAL_GAMES = 5;

// Weight given to championship % (peak performance) vs. average pairwise win
// probability (consistent dominance) in SIM SCORE. 0 = pure pairwise dominance,
// 1 = pure championship %. Low default because championship % is high-variance
// at small game-pool sizes.
export const SIMULATION_SCORE_ALPHA = 0.3;

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

// Composite SIM SCORE: blends championship % (peak performance, normalized
// against the "fair share" 1/N a uniformly-skilled player would win) with
// average pairwise win probability (consistent dominance across the pool).
// Pure function of already-computed per-player numbers, so it can be
// recomputed client-side for a different alpha without rerunning the simulation.
export function simulationScore(
  champPct: number,
  avgPairwiseWinProb: number,
  nPlayers: number,
  alpha: number = SIMULATION_SCORE_ALPHA,
): number {
  const fairShare = nPlayers > 0 ? 1 / nPlayers : 0;
  const champNorm = fairShare > 0 ? Math.min(1, champPct / fairShare) : 0;
  return alpha * champNorm + (1 - alpha) * avgPairwiseWinProb;
}

export function simulateRoundRobin(state: GraphState, engine?: UnifiedEngine): SimulationResult {
  const players = Object.values(state.players);
  if (players.length === 0) return { ranking: [], h2h_sim: {}, champ_counts: {} };
  const n = players.length;

  // Build Katz matrix for fallback predictions when no engine is provided
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
      const p_a_wins = engine
        ? engine.predict(a.id, b.id).p_a_wins
        : katzPredict(katz, a.id, b.id, 0).p_a_wins;
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

  const pageRank = computePageRank(state);
  const fairShare = 1 / n;

  const ranking: RankEntry[] = players
    .map(p => {
      const champPct = champCount[p.id] / SIM_ROUNDS;
      const avgPairwiseWinProb = n > 1
        ? players.filter(q => q.id !== p.id).reduce((sum, q) => sum + h2h[p.id][q.id] / SIM_ROUNDS, 0) / (n - 1)
        : 0;
      const avgPlacement = placementCounts[p.id].reduce((sum, count, k) => sum + count * (k + 1), 0) / SIM_ROUNDS;
      return {
        player_id:          p.id,
        display_name:       p.display_name,
        tournament_wins:    champCount[p.id],
        tournament_win_pct: champPct,
        placement_dist:     placementCounts[p.id],
        stat_vec:           statVecs[p.id],
        matchup_table:      matchupTable[p.id],
        page_rank:          pageRank[p.id] ?? 0,
        avg_pairwise_win_prob: avgPairwiseWinProb,
        avg_placement:         avgPlacement,
        champ_pct_norm:        Math.min(1, champPct / fairShare),
        sim_score:             simulationScore(champPct, avgPairwiseWinProb, n, SIMULATION_SCORE_ALPHA),
      };
    })
    .sort((a, b) =>
      b.sim_score - a.sim_score ||
      a.avg_placement - b.avg_placement
    );

  return { ranking, h2h_sim: h2h, champ_counts: champCount };
}

// ─── ELO ─────────────────────────────────────────────────────────────────────

export function computeElo(state: GraphState): Record<string, number> {
  const elo: Record<string, number> = {};
  const gc: Record<string, number> = {};
  for (const id of Object.keys(state.players)) { elo[id] = ELO_START; gc[id] = 0; }
  for (const g of [...state.games].sort((a, b) => a.timestamp - b.timestamp)) {
    const ra = elo[g.winner_id] ?? ELO_START, rb = elo[g.loser_id] ?? ELO_START;
    const kW = (gc[g.winner_id] ?? 0) < PROVISIONAL_GAMES ? K_PROVISIONAL : K_STABLE;
    const kL = (gc[g.loser_id]  ?? 0) < PROVISIONAL_GAMES ? K_PROVISIONAL : K_STABLE;
    const ea = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    elo[g.winner_id] = Math.round(ra + kW * (1 - ea));
    elo[g.loser_id]  = Math.round(rb + kL * (0 - (1 - ea)));
    gc[g.winner_id] = (gc[g.winner_id] ?? 0) + 1;
    gc[g.loser_id]  = (gc[g.loser_id]  ?? 0) + 1;
  }
  return elo;
}

export interface EloHistoryPoint {
  timestamp: number;
  elo: number;
  opponent_id: string;
  result: "W" | "L";
  score_for?: number;
  score_against?: number;
  delta: number;
}

export function computeEloHistory(state: GraphState): Record<string, EloHistoryPoint[]> {
  const current: Record<string, number> = {};
  const gc: Record<string, number> = {};
  const history: Record<string, EloHistoryPoint[]> = {};
  for (const id of Object.keys(state.players)) { current[id] = ELO_START; gc[id] = 0; history[id] = []; }
  for (const g of [...state.games].sort((a, b) => a.timestamp - b.timestamp)) {
    const ra = current[g.winner_id] ?? ELO_START, rb = current[g.loser_id] ?? ELO_START;
    const kW = (gc[g.winner_id] ?? 0) < PROVISIONAL_GAMES ? K_PROVISIONAL : K_STABLE;
    const kL = (gc[g.loser_id]  ?? 0) < PROVISIONAL_GAMES ? K_PROVISIONAL : K_STABLE;
    const ea = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    current[g.winner_id] = Math.round(ra + kW * (1 - ea));
    current[g.loser_id]  = Math.round(rb + kL * (0 - (1 - ea)));
    gc[g.winner_id] = (gc[g.winner_id] ?? 0) + 1;
    gc[g.loser_id]  = (gc[g.loser_id]  ?? 0) + 1;
    if (history[g.winner_id]) history[g.winner_id].push({
      timestamp: g.timestamp, elo: current[g.winner_id], opponent_id: g.loser_id, result: "W",
      score_for: g.score_winner, score_against: g.score_loser, delta: current[g.winner_id] - ra,
    });
    if (history[g.loser_id]) history[g.loser_id].push({
      timestamp: g.timestamp, elo: current[g.loser_id], opponent_id: g.winner_id, result: "L",
      score_for: g.score_loser, score_against: g.score_winner, delta: current[g.loser_id] - rb,
    });
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

    if (myGames.length === 0) { result[id] = ELO_START; continue; }

    // Build mini-state with only the players in those games
    const miniPlayers: Record<string, { id: string; display_name: string; created_at: number }> = {};
    for (const g of myGames) {
      if (state.players[g.winner_id]) miniPlayers[g.winner_id] = state.players[g.winner_id];
      if (state.players[g.loser_id])  miniPlayers[g.loser_id]  = state.players[g.loser_id];
    }
    const miniElo = computeElo({ players: miniPlayers, games: myGames });
    result[id] = miniElo[id] ?? ELO_START;
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
  const rA = eloAtLast[aId] ?? ELO_START, rB = eloAtLast[bId] ?? ELO_START;
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

// ─── Extended Player Stats ────────────────────────────────────────────────────

export interface ExtendedPlayerStats {
  kill_rate: number;              // kills / total rounds played
  round_win_rate: number;         // rounds won / total rounds played
  efficiency: number;             // round_win_rate - kill_rate
  aggression: number;             // kill_rate - death_rate (per round)
  upward_wr: number | null;       // WR vs players ranked above
  downward_wr: number | null;     // WR vs players ranked below
  strength_of_schedule: number;  // avg opponent ELO weighted by games
  peak_elo: number;
  elo_volatility: number;         // std dev of per-game ELO deltas
  predictability_entropy: number; // Shannon entropy of outcome distribution (0–1)
  has_score_data: boolean;
}

export function computeExtendedPlayerStats(
  state: GraphState,
  playerId: string,
  eloHistory: Array<{ timestamp: number; elo: number }>,
  playerRank: number,            // 0-indexed rank of this player
): ExtendedPlayerStats {
  const myGames = state.games.filter(g => g.winner_id === playerId || g.loser_id === playerId);
  const elo = computeElo(state);
  const ids = Object.keys(state.players);
  const eloOrder = [...ids].sort((a, b) => (elo[b] ?? 1000) - (elo[a] ?? 1000));

  let totalKills = 0, totalDeaths = 0, totalRoundsPlayed = 0, totalRoundsWon = 0;
  let hasScore = false;
  for (const g of myGames) {
    const won = g.winner_id === playerId;
    const s = won ? g.winner_stats : g.loser_stats;
    totalKills += s.kills;
    totalDeaths += s.deaths;
    if (g.score_winner !== undefined && g.score_loser !== undefined) {
      hasScore = true;
      const my = won ? g.score_winner : g.score_loser;
      const op = won ? g.score_loser : g.score_winner;
      totalRoundsPlayed += my + op;
      totalRoundsWon += my;
    }
  }
  const kill_rate = totalRoundsPlayed > 0 ? totalKills / totalRoundsPlayed : 0;
  const death_rate = totalRoundsPlayed > 0 ? totalDeaths / totalRoundsPlayed : 0;
  const round_win_rate = totalRoundsPlayed > 0 ? totalRoundsWon / totalRoundsPlayed : 0;

  // Upward / downward WR
  let upWins = 0, upGames = 0, downWins = 0, downGames = 0;
  for (const oppId of ids) {
    if (oppId === playerId) continue;
    const oppRank = eloOrder.indexOf(oppId);
    const h2hWins   = myGames.filter(g => g.winner_id === playerId && g.loser_id === oppId).length;
    const h2hLosses = myGames.filter(g => g.loser_id  === playerId && g.winner_id === oppId).length;
    const total = h2hWins + h2hLosses;
    if (total === 0) continue;
    if (oppRank < playerRank) { upWins += h2hWins; upGames += total; }
    else                      { downWins += h2hWins; downGames += total; }
  }

  // Strength of schedule
  let sosSum = 0, sosCount = 0;
  for (const g of myGames) {
    const oppId = g.winner_id === playerId ? g.loser_id : g.winner_id;
    sosSum += elo[oppId] ?? 1000;
    sosCount++;
  }

  // Peak ELO
  const peak_elo = eloHistory.length > 0 ? Math.max(...eloHistory.map(h => h.elo)) : 1000;

  // ELO volatility
  let elo_volatility = 0;
  if (eloHistory.length >= 2) {
    const deltas = eloHistory.slice(1).map((h, i) => h.elo - eloHistory[i].elo);
    const dmean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    elo_volatility = Math.sqrt(deltas.reduce((s, d) => s + (d - dmean) ** 2, 0) / deltas.length);
  }

  // Predictability entropy: Shannon entropy of win/loss pattern per opponent
  let entropy = 0;
  for (const oppId of ids) {
    if (oppId === playerId) continue;
    const h2h = myGames.filter(g =>
      (g.winner_id === playerId && g.loser_id === oppId) ||
      (g.loser_id  === playerId && g.winner_id === oppId)
    );
    if (h2h.length === 0) continue;
    const w = h2h.filter(g => g.winner_id === playerId).length;
    const p = w / h2h.length, q = 1 - p;
    const weight = h2h.length / Math.max(myGames.length, 1);
    if (p > 0) entropy -= weight * p * Math.log2(p);
    if (q > 0) entropy -= weight * q * Math.log2(q);
  }

  return {
    kill_rate:               Math.round(kill_rate  * 1000) / 1000,
    round_win_rate:          Math.round(round_win_rate * 1000) / 1000,
    efficiency:              Math.round((round_win_rate - kill_rate) * 1000) / 1000,
    aggression:              Math.round((kill_rate - death_rate)     * 1000) / 1000,
    upward_wr:               upGames   > 0 ? Math.round((upWins   / upGames)   * 100) / 100 : null,
    downward_wr:             downGames > 0 ? Math.round((downWins / downGames) * 100) / 100 : null,
    strength_of_schedule:    sosCount  > 0 ? Math.round(sosSum / sosCount) : 1000,
    peak_elo,
    elo_volatility:          Math.round(elo_volatility),
    predictability_entropy:  Math.min(1, Math.round(entropy * 100) / 100),
    has_score_data:          hasScore,
  };
}

// ─── Dominance Duration ───────────────────────────────────────────────────────

export interface DominanceDuration {
  current_rank: number;           // 1-indexed
  days_at_rank: number;
  games_at_rank: number;
  peak_rank: number;              // 1-indexed best rank ever
  rank_trajectory: number[];      // last 10 ranks (1-indexed)
}

export function computeDominanceDuration(
  state: GraphState,
  playerId: string,
): DominanceDuration | null {
  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  if (sorted.length === 0) return null;
  const ids = Object.keys(state.players);

  const rankHistory: Array<{ rank: number; timestamp: number }> = [];
  for (let i = 1; i <= sorted.length; i++) {
    const snap = computeElo({ players: state.players, games: sorted.slice(0, i) });
    const order = [...ids].sort((a, b) => (snap[b] ?? 1000) - (snap[a] ?? 1000));
    const rank = order.indexOf(playerId);
    if (rank !== -1) rankHistory.push({ rank, timestamp: sorted[i - 1].timestamp });
  }
  if (rankHistory.length === 0) return null;

  const currentRank = rankHistory[rankHistory.length - 1].rank;
  const peakRank    = Math.min(...rankHistory.map(h => h.rank));

  let daysAtRank = 0, gamesAtRank = 0;
  for (let i = rankHistory.length - 1; i >= 0; i--) {
    if (rankHistory[i].rank !== currentRank) break;
    gamesAtRank++;
    if (i > 0) daysAtRank += (rankHistory[i].timestamp - rankHistory[i - 1].timestamp) / 86_400_000;
  }

  return {
    current_rank:     currentRank + 1,
    days_at_rank:     Math.round(daysAtRank),
    games_at_rank:    gamesAtRank,
    peak_rank:        peakRank + 1,
    rank_trajectory:  rankHistory.slice(-10).map(h => h.rank + 1),
  };
}

// ─── Scoreline Probability Model (Negative Binomial) ──────────────────────────
// A "match" is first-to-MATCH_TARGET rounds. These convert between match-level
// win probability and per-round win rate, and enumerate every possible final
// scoreline with its probability.

const MATCH_TARGET = 7;

export function binomialCoeff(n: number, k: number): number {
  if (k === 0 || k === n) return 1;
  if (k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i++) result *= (n - i) / (i + 1);
  return Math.round(result);
}

// Per-round win rate p -> probability of winning a first-to-`target` match.
export function matchWinProb(p: number, target: number = MATCH_TARGET): number {
  let prob = 0;
  for (let k = target; k <= 2 * target - 1; k++) {
    prob += binomialCoeff(k - 1, target - 1) * Math.pow(p, target) * Math.pow(1 - p, k - target);
  }
  return prob;
}

// Inverse of matchWinProb (bisection): match win probability -> per-round win rate.
export function roundWinRateFromMatchProb(P: number, target: number = MATCH_TARGET, tolerance: number = 1e-8): number {
  if (Math.abs(P - 0.5) < tolerance) return 0.5;
  let lo = 0.001, hi = 0.999;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    const prob = matchWinProb(mid, target);
    if (Math.abs(prob - P) < tolerance) return mid;
    if (prob < P) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

export interface ScorelineOutcome {
  scoreA: number;
  scoreB: number;
  prob: number;
}

// Probability of a specific (scoreA, scoreB) scoreline given A's per-round win
// rate p. The winner clinches the final round having won (target-1) of the
// (total-1) rounds before it — a negative binomial.
export function scorelineProbability(scoreA: number, scoreB: number, p: number, target: number = MATCH_TARGET): number {
  const aWins = scoreA === target;
  const pWinner = aWins ? p : 1 - p;
  const loserScore = aWins ? scoreB : scoreA;
  const totalRounds = target + loserScore;
  return binomialCoeff(totalRounds - 1, target - 1) *
         Math.pow(pWinner, target) *
         Math.pow(1 - pWinner, loserScore);
}

// All 13 possible first-to-`target` scorelines with their probabilities (sum to 1).
export function allScorelineProbabilities(p: number, target: number = MATCH_TARGET): ScorelineOutcome[] {
  const outcomes: ScorelineOutcome[] = [];
  for (let loser = 0; loser < target; loser++) {
    outcomes.push({ scoreA: target, scoreB: loser, prob: scorelineProbability(target, loser, p, target) });
    outcomes.push({ scoreA: loser, scoreB: target, prob: scorelineProbability(loser, target, p, target) });
  }
  return outcomes;
}

// ─── Information Gain Recommendation ─────────────────────────────────────────
// Exact: enumerate every possible scoreline for a candidate match, weight each
// by its probability under the current model, refit BT under each hypothetical
// outcome, and take the probability-weighted average uncertainty reduction.
// Unplayed pairs naturally outscore rematches — a first direct game collapses a
// large amount of transitive uncertainty, which falls straight out of the math.

export interface InfoGainEntry {
  a: string;
  b: string;
  info_gain: number;
  direct_games: number;
  win_prob: number;               // p(a beats b) from BT
  label: "VERY HIGH" | "HIGH" | "MED" | "LOW";
  context: string;
}

// Total model uncertainty: sum of squared standard errors across all BT ratings.
function totalUncertainty(bt: BTResult): number {
  return Object.values(bt.se).reduce((sum, se) => sum + se * se, 0);
}

function exactInfoGain(state: GraphState, aId: string, bId: string, uCurrent: number, matchProb: number): number {
  const p = roundWinRateFromMatchProb(Math.min(0.999, Math.max(0.001, matchProb)));

  let expectedAfter = 0;
  for (const { scoreA, scoreB, prob } of allScorelineProbabilities(p)) {
    if (prob < 1e-6) continue; // negligible outcome — skip the BT refit

    const aWins = scoreA === MATCH_TARGET;
    const synGame: Game = {
      id: "synthetic",
      timestamp: Date.now(),
      winner_id: aWins ? aId : bId,
      loser_id:  aWins ? bId : aId,
      winner_stats: { kills: 0, deaths: 0 },
      loser_stats:  { kills: 0, deaths: 0 },
      score_winner: MATCH_TARGET,
      score_loser:  aWins ? scoreB : scoreA,
    };

    const hypoBT = computeBradleyTerry({ players: state.players, games: [...state.games, synGame] });
    if (!hypoBT) continue;
    expectedAfter += prob * totalUncertainty(hypoBT);
  }

  return uCurrent - expectedAfter;
}

export function computeInformationGain(state: GraphState, bt: BTResult, h2hSim: H2hSim): InfoGainEntry[] {
  const ids = Object.keys(state.players);
  const n = ids.length;
  if (n < 2) return [];

  const uCurrent = totalUncertainty(bt);

  const candidates: InfoGainEntry[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const ai = ids[i], bi = ids[j];
      const betaA = bt.beta[ai] ?? 0, betaB = bt.beta[bi] ?? 0;
      const p = 1 / (1 + Math.exp(betaB - betaA));

      const matchProb = (h2hSim[ai]?.[bi] ?? SIM_ROUNDS / 2) / SIM_ROUNDS;
      const info_gain = exactInfoGain(state, ai, bi, uCurrent, matchProb);

      const directGames = state.games.filter(g =>
        (g.winner_id === ai && g.loser_id === bi) ||
        (g.winner_id === bi && g.loser_id === ai)
      ).length;

      let context: string;
      if (directGames === 0) {
        context = "First ever meeting — any result teaches the model a lot";
      } else {
        const lastH2H = [...state.games]
          .filter(g => (g.winner_id === ai && g.loser_id === bi) || (g.winner_id === bi && g.loser_id === ai))
          .sort((a, b) => b.timestamp - a.timestamp)[0];
        let drift = 0;
        if (lastH2H) {
          const eloThen = computeElo({ players: state.players, games: state.games.filter(g => g.timestamp <= lastH2H.timestamp) });
          const pThen = 1 / (1 + Math.pow(10, ((eloThen[bi] ?? 1000) - (eloThen[ai] ?? 1000)) / 400));
          drift = Math.abs(p - pThen);
        }
        context = drift > 0.15
          ? "Ratings have shifted significantly since last meeting"
          : `High model uncertainty about this matchup · ${directGames} prior game${directGames > 1 ? 's' : ''}`;
      }

      candidates.push({ a: ai, b: bi, info_gain, direct_games: directGames, win_prob: p, label: "LOW", context });
    }
  }

  candidates.sort((a, b) => b.info_gain - a.info_gain);
  const maxG = candidates[0]?.info_gain ?? 1;
  for (const c of candidates) {
    const norm = maxG > 0 ? c.info_gain / maxG : 0;
    c.label = norm > 0.75 ? "VERY HIGH" : norm > 0.5 ? "HIGH" : norm > 0.25 ? "MED" : "LOW";
  }
  return candidates;
}

// ─── Retroactive Game Impact ──────────────────────────────────────────────────
// Impact = reduction in total BT variance (Σ SE²) when this game was added.

export function computeRetroactiveImpact(state: GraphState): Record<string, number> {
  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  const n = Object.keys(state.players).length;
  const DEFAULT_SE = 200;

  function totalVar(games: Game[]): number {
    if (games.length < 2) return n * DEFAULT_SE ** 2;
    const bt = computeBradleyTerry({ players: state.players, games });
    if (!bt) return n * DEFAULT_SE ** 2;
    return Object.keys(state.players).reduce((s, id) => s + (bt.se[id] ?? DEFAULT_SE) ** 2, 0);
  }

  const impacts: number[] = sorted.map((_, i) => {
    const before = totalVar(sorted.slice(0, i));
    const after  = totalVar(sorted.slice(0, i + 1));
    return Math.max(0, before - after);
  });

  const maxImpact = Math.max(...impacts, 1);
  const result: Record<string, number> = {};
  sorted.forEach((g, i) => { result[g.id] = impacts[i] / maxImpact; });
  return result;
}

// ─── Kingmaker / Leave-One-Out Sensitivity ────────────────────────────────────

export interface KingmakerEntry {
  player_id: string;
  display_name: string;
  kendall_tau_shift: number;      // 1 - tau (higher = more structurally important)
  max_position_shift: number;
}

export function computeKingmaker(state: GraphState): KingmakerEntry[] {
  const ids = Object.keys(state.players);
  if (ids.length < 3) return [];

  const fullElo = computeElo(state);
  const fullOrder = [...ids].sort((a, b) => (fullElo[b] ?? 1000) - (fullElo[a] ?? 1000));

  return ids.map(removedId => {
    const remaining = ids.filter(id => id !== removedId);
    const redPlayers: Record<string, Player> = {};
    for (const id of remaining) { if (state.players[id]) redPlayers[id] = state.players[id]; }
    const redGames = state.games.filter(g => g.winner_id !== removedId && g.loser_id !== removedId);
    const redElo = computeElo({ players: redPlayers, games: redGames });
    const redOrder = [...remaining].sort((a, b) => (redElo[b] ?? 1000) - (redElo[a] ?? 1000));

    const fullWithout = fullOrder.filter(id => id !== removedId);
    const tau = kendallTauAgreement(fullWithout, redOrder);
    const maxShift = remaining.reduce((mx, id) => {
      const before = fullWithout.indexOf(id), after = redOrder.indexOf(id);
      return before !== -1 && after !== -1 ? Math.max(mx, Math.abs(after - before)) : mx;
    }, 0);

    return {
      player_id:          removedId,
      display_name:       state.players[removedId]?.display_name ?? removedId,
      kendall_tau_shift:  Math.round((1 - tau) * 100) / 100,
      max_position_shift: maxShift,
    };
  }).sort((a, b) => b.kendall_tau_shift - a.kendall_tau_shift);
}

// ─── Pool Calibration Curve ───────────────────────────────────────────────────

export interface CalibrationBucket {
  predicted_mid: number;  // center of probability bucket
  actual_rate: number;    // fraction where higher-ELO player won
  n: number;
}

export function computeCalibrationCurve(state: GraphState): CalibrationBucket[] {
  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  const buckets: { sum: number; n: number }[] = Array.from({ length: 4 }, () => ({ sum: 0, n: 0 }));
  const edges = [0.5, 0.6, 0.7, 0.8, 1.0];

  for (let i = 2; i < sorted.length; i++) {
    const g = sorted[i];
    const hist = computeElo({ players: state.players, games: sorted.slice(0, i) });
    const rW = hist[g.winner_id] ?? 1000, rL = hist[g.loser_id] ?? 1000;
    const rHigh = Math.max(rW, rL);
    const pred = 1 / (1 + Math.pow(10, ((rHigh === rW ? rL : rW) - rHigh) / 400));
    const actual = rW >= rL ? 1 : 0;
    const bi = edges.findIndex((e, i) => i < edges.length - 1 && pred >= edges[i] && pred < edges[i + 1]);
    if (bi >= 0) { buckets[bi].sum += actual; buckets[bi].n++; }
  }

  return buckets
    .map((b, i) => ({ predicted_mid: (edges[i] + edges[i + 1]) / 2, actual_rate: b.n > 0 ? b.sum / b.n : 0, n: b.n }))
    .filter(b => b.n > 0);
}

// ─── Permutation Test ─────────────────────────────────────────────────────────

export interface PermutationTestResult {
  p_value: number;
  significant: boolean;
  label: string;
}

export function computePermutationTest(state: GraphState, nPermutations = 200): PermutationTestResult | null {
  if (state.games.length < 6 || Object.keys(state.players).length < 3) return null;
  const bt = computeBradleyTerry(state);
  if (!bt) return null;

  function btLogLik(games: Game[], beta: Record<string, number>): number {
    return games.reduce((ll, g) => {
      const p = 1 / (1 + Math.exp((beta[g.loser_id] ?? 0) - (beta[g.winner_id] ?? 0)));
      return ll + Math.log(Math.max(1e-10, p));
    }, 0);
  }

  const realLL = btLogLik(state.games, bt.beta);
  let moreExtreme = 0;
  for (let p = 0; p < nPermutations; p++) {
    const shuffled: Game[] = state.games.map(g =>
      Math.random() < 0.5
        ? { ...g, winner_id: g.loser_id, loser_id: g.winner_id, winner_stats: g.loser_stats, loser_stats: g.winner_stats }
        : g
    );
    const pb = computeBradleyTerry({ players: state.players, games: shuffled });
    if (pb && btLogLik(shuffled, pb.beta) >= realLL) moreExtreme++;
  }

  const pv = moreExtreme / nPermutations;
  return {
    p_value:     Math.round(pv * 100) / 100,
    significant: pv < 0.05,
    label:       pv < 0.05
      ? `SIGNIFICANT (p=${pv.toFixed(2)}) — rankings reflect real skill structure`
      : `NOT SIGNIFICANT (p=${pv.toFixed(2)}) — insufficient data to confirm real signal`,
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

// ─── Ordinal Bradley-Terry (margin-aware) ─────────────────────────────────────
// Models the round differential d = score_A - score_B as an ordinal variable
// under a logistic CDF with 13 derived cutpoints. Thresholds are computed
// analytically from the first-to-7 negative-binomial PMF (no free threshold
// params). Only fitted parameters: β per player (zero-mean skill) + log σ (scale).

const ORDINAL_DIFFS = [-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7] as const;

export interface OrdinalBTFit {
  beta: Record<string, number>;  // zero-mean log-skill
  sigma: number;                 // scale (fitted)
  cutpoints: number[];           // 13 logit-scale cutpoints
  se: Record<string, number>;    // per-player SE from diagonal Hessian
  ids: string[];
}

function logisticFn(x: number): number { return 1 / (1 + Math.exp(-x)); }

function deriveOrdinalCutpoints(): number[] {
  let cumP = 0;
  return (ORDINAL_DIFFS.slice(0, -1) as number[]).map(d => {
    const prob = d < 0
      ? scorelineProbability(MATCH_TARGET + d, MATCH_TARGET, 0.5)
      : scorelineProbability(MATCH_TARGET, MATCH_TARGET - d, 0.5);
    cumP += prob;
    const p = Math.max(1e-9, Math.min(1 - 1e-9, cumP));
    return Math.log(p / (1 - p));
  });
}

function ordinalBTPointLik(d: number, loc: number, sigma: number, cutpoints: number[]): number {
  const k = (ORDINAL_DIFFS as readonly number[]).indexOf(d);
  if (k < 0) return 1e-10;
  const invS = 1 / sigma;
  const upper = k < cutpoints.length ? logisticFn((cutpoints[k] - loc) * invS) : 1.0;
  const lower = k > 0 ? logisticFn((cutpoints[k - 1] - loc) * invS) : 0.0;
  return Math.max(1e-10, upper - lower);
}

export function ordinalBTWinProb(betaA: number, betaB: number, sigma: number, cutpoints: number[]): number {
  const loc = betaA - betaB;
  let p = 0;
  for (const d of [1,2,3,4,5,6,7]) p += ordinalBTPointLik(d, loc, sigma, cutpoints);
  return p;
}

export function ordinalBTPredictedMargin(betaA: number, betaB: number, sigma: number, cutpoints: number[]): number {
  const loc = betaA - betaB;
  return (ORDINAL_DIFFS as readonly number[]).reduce(
    (exp, d) => exp + d * ordinalBTPointLik(d, loc, sigma, cutpoints), 0
  );
}

export function fitOrdinalBT(state: GraphState, nowMs = Date.now()): OrdinalBTFit | null {
  const ids = Object.keys(state.players);
  const n = ids.length;
  if (n < 2) return null;

  const scored = state.games
    .filter(g => g.score_winner !== undefined && g.score_loser !== undefined)
    .sort((a, b) => a.timestamp - b.timestamp);
  if (scored.length < 2) return null;

  const idx: Record<string, number> = {};
  ids.forEach((id, i) => { idx[id] = i; });

  const cutpoints = deriveOrdinalCutpoints();
  const recency = scored.map(g => Math.exp(-(nowMs - g.timestamp) / (TAU_DAYS * 86_400_000)));

  const gc: Record<string, number> = {};
  for (const id of ids) gc[id] = 0;
  for (const g of scored) {
    gc[g.winner_id] = (gc[g.winner_id] ?? 0) + 1;
    gc[g.loser_id]  = (gc[g.loser_id]  ?? 0) + 1;
  }

  // params: [beta_0..beta_{n-1}, log_sigma]
  const params = new Array(n + 1).fill(0);

  function objective(p: number[]): number {
    const raw = p.slice(0, n);
    const mean = raw.reduce((a, b) => a + b, 0) / n;
    const betas = raw.map(b => b - mean);
    const sigma = Math.exp(Math.max(-3, Math.min(3, p[n])));
    let ll = 0;
    for (let gi = 0; gi < scored.length; gi++) {
      const g = scored[gi];
      const wi = idx[g.winner_id], li = idx[g.loser_id];
      if (wi === undefined || li === undefined) continue;
      const d = g.score_winner! - g.score_loser!;
      if (d <= 0) continue;
      ll += recency[gi] * Math.log(ordinalBTPointLik(d, betas[wi] - betas[li], sigma, cutpoints));
    }
    for (let i = 0; i < n; i++) {
      const lambda = 0.5 * Math.exp(-(gc[ids[i]] ?? 0) / 4);
      ll -= lambda * betas[i] * betas[i] / 2;
    }
    return ll;
  }

  const h = 1e-5, lr = 0.08, MAX_ITER = 600, TOL = 1e-7;
  for (let iter = 0; iter < MAX_ITER; iter++) {
    const grad = new Array(n + 1).fill(0);
    for (let k = 0; k <= n; k++) {
      params[k] += h; const up = objective(params);
      params[k] -= 2 * h; const dn = objective(params);
      params[k] += h;
      grad[k] = (up - dn) / (2 * h);
    }
    let stepNorm = 0;
    for (let k = 0; k <= n; k++) { params[k] += lr * grad[k]; stepNorm += grad[k] * grad[k]; }
    if (Math.sqrt(stepNorm) * lr < TOL) break;
  }

  const rawBetas = params.slice(0, n);
  const finalMean = rawBetas.reduce((a, b) => a + b, 0) / n;
  const finalBetas = rawBetas.map(b => b - finalMean);
  const sigma = Math.exp(Math.max(-3, Math.min(3, params[n])));

  const h2 = 1e-4, f0 = objective(params);
  const beta: Record<string, number> = {};
  const se: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    beta[ids[i]] = finalBetas[i];
    params[i] += h2; const up = objective(params);
    params[i] -= 2 * h2; const dn = objective(params);
    params[i] += h2;
    const hii = (up - 2 * f0 + dn) / (h2 * h2);
    se[ids[i]] = hii < -1e-9 ? Math.sqrt(1 / (-hii)) : 1.0;
  }

  return { beta, sigma, cutpoints, se, ids };
}

// ─── Unified Prediction Engine ────────────────────────────────────────────────
// Two-layer ensemble: ordinal BT (margin-aware) + LOO-fit logistic regression
// blending ELO / Katz / ordinal BT signals. Features use log-odds (logit) for
// probability inputs so equal default weights produce a correctly-centred prior.

const ENSEMBLE_N_FEATURES = 6;
const ENSEMBLE_DEFAULT_WEIGHTS = [0.5, 0.2, 0.5, 0.1, 0.3, 0.0]; // ELO logit, Katz z, BT logit, BT reliability z, H2H logit, intercept
const ENSEMBLE_MIN_LOO = 4;

export interface UnifiedEngineStatus {
  ensembleWeights: number[];
  ensembleMode: "FITTED" | "DEFAULTED" | "TOO_FEW_GAMES";
  looN: number;
  looBrierFitted: number | null;
  looBrierDefault: number | null;
  ordinalBTFitted: boolean;
  sigma: number | null;
}

export interface UnifiedEngine {
  predict: (aId: string, bId: string) => PairwisePrediction;
  ordinalBT: OrdinalBTFit | null;
  status: UnifiedEngineStatus;
}

function ensembleSig(x: number): number { return 1 / (1 + Math.exp(-x)); }
function ensembleDot(a: number[], b: number[]): number { return a.reduce((s, v, i) => s + v * b[i], 0); }
function logitClamp(p: number): number {
  const pc = Math.max(0.02, Math.min(0.98, p));
  return Math.log(pc / (1 - pc));
}

function katzCentrality(K: number[][], n: number): number[] {
  return Array.from({ length: n }, (_, i) =>
    K[i].reduce((s, v, j) => (i === j ? s : s + Math.max(0, v)), 0)
  );
}

function engineFeatures(
  aId: string, bId: string,
  eloMap: Record<string, number>,
  katzR: KatzResult,
  katzStd: number,
  obt: OrdinalBTFit | null,
  h2hGames: Game[],   // games between aId and bId from the training set
): number[] {
  const eloA = eloMap[aId] ?? ELO_START, eloB = eloMap[bId] ?? ELO_START;
  const eloP = 1 / (1 + Math.pow(10, (eloB - eloA) / 400));

  const { idx, K } = katzR;
  const nk = katzR.ids.length;
  const cent = katzCentrality(K, nk);
  const ai = idx[aId] ?? -1, bi = idx[bId] ?? -1;
  const katzZ = katzStd > 1e-9
    ? ((ai >= 0 ? cent[ai] : 0) - (bi >= 0 ? cent[bi] : 0)) / katzStd
    : 0;

  let btLogit = logitClamp(eloP);
  let btRel = 0;
  if (obt && obt.beta[aId] !== undefined && obt.beta[bId] !== undefined) {
    const bA = obt.beta[aId], bB = obt.beta[bId];
    const sA = obt.se[aId] ?? 1.0, sB = obt.se[bId] ?? 1.0;
    btLogit = logitClamp(ordinalBTWinProb(bA, bB, obt.sigma, obt.cutpoints));
    btRel = (bA - bB) / Math.sqrt(sA * sA + sB * sB + 1e-6);
  }

  // H2H observed win rate, Jeffreys-smoothed (adds 0.5 pseudo-counts per side).
  // Zero when no H2H games — contributes nothing to the prior.
  const h2hN = h2hGames.length;
  const h2hAWins = h2hGames.filter(g => g.winner_id === aId).length;
  const h2hLogit = h2hN > 0 ? logitClamp((h2hAWins + 0.5) / (h2hN + 1)) : 0;

  return [logitClamp(eloP), katzZ, btLogit, btRel, h2hLogit, 1.0];
}

function fitEnsembleWeights(examples: Array<{ x: number[]; y: number }>): number[] {
  let w = ENSEMBLE_DEFAULT_WEIGHTS.slice();
  const lr = 0.05, l2 = 1.0, MAX_ITER = 1000, TOL = 1e-8;
  for (let iter = 0; iter < MAX_ITER; iter++) {
    const grad = new Array(ENSEMBLE_N_FEATURES).fill(0);
    for (const { x, y } of examples) {
      const r = ensembleSig(ensembleDot(w, x)) - y;
      for (let j = 0; j < ENSEMBLE_N_FEATURES; j++) grad[j] += r * x[j];
    }
    for (let j = 0; j < ENSEMBLE_N_FEATURES - 1; j++) grad[j] += l2 * w[j];
    let maxChange = 0;
    for (let j = 0; j < ENSEMBLE_N_FEATURES; j++) {
      const step = lr * grad[j] / examples.length;
      w[j] -= step;
      maxChange = Math.max(maxChange, Math.abs(step));
    }
    if (maxChange < TOL) break;
  }
  return w;
}

function ensembleBrier(examples: Array<{ x: number[]; y: number }>, w: number[]): number {
  if (examples.length === 0) return 0.25;
  return examples.reduce((s, { x, y }) => s + (ensembleSig(ensembleDot(w, x)) - y) ** 2, 0) / examples.length;
}

function weightsSane(w: number[]): boolean {
  return w[0] >= -0.5 && w[2] >= -0.5 && w[4] >= -0.5 && Math.abs(w[5]) <= 2.0;
}

export function buildUnifiedEngine(state: GraphState, nowMs = Date.now()): UnifiedEngine {
  const eloMap = computeElo(state);
  const katz = buildKatzMatrix(state);
  const obt = fitOrdinalBT(state, nowMs);

  const nk = katz.ids.length;
  const cent = katzCentrality(katz.K, nk);
  const kMean = cent.reduce((a, b) => a + b, 0) / Math.max(1, nk);
  const katzStd = Math.sqrt(cent.reduce((s, v) => s + (v - kMean) ** 2, 0) / Math.max(1, nk)) || 1;

  const sorted = [...state.games].sort((a, b) => a.timestamp - b.timestamp);
  const looEx: Array<{ x: number[]; y: number }> = [];

  if (sorted.length >= ENSEMBLE_MIN_LOO) {
    for (let k = 0; k < sorted.length; k++) {
      const g = sorted[k];
      if (!state.players[g.winner_id] || !state.players[g.loser_id]) continue;
      const ts: GraphState = { players: state.players, games: sorted.filter((_, i) => i !== k) };
      const lElo  = computeElo(ts);
      const lKatz = buildKatzMatrix(ts);
      const lObt  = fitOrdinalBT(ts, nowMs);
      const lc = katzCentrality(lKatz.K, lKatz.ids.length);
      const lMean = lc.reduce((a, b) => a + b, 0) / Math.max(1, lKatz.ids.length);
      const lStd  = Math.sqrt(lc.reduce((s, v) => s + (v - lMean) ** 2, 0) / Math.max(1, lKatz.ids.length)) || 1;
      const lH2H = ts.games.filter(tg =>
        (tg.winner_id === g.winner_id && tg.loser_id === g.loser_id) ||
        (tg.winner_id === g.loser_id && tg.loser_id === g.winner_id)
      );
      looEx.push({ x: engineFeatures(g.winner_id, g.loser_id, lElo, lKatz, lStd, lObt, lH2H), y: 1 });
    }
  }

  let weights = ENSEMBLE_DEFAULT_WEIGHTS.slice();
  let mode: "FITTED" | "DEFAULTED" | "TOO_FEW_GAMES" = "TOO_FEW_GAMES";
  let brierFitted: number | null = null, brierDefault: number | null = null;

  if (looEx.length >= ENSEMBLE_MIN_LOO) {
    const fitted = fitEnsembleWeights(looEx);
    const bF = ensembleBrier(looEx, fitted);
    const bD = ensembleBrier(looEx, ENSEMBLE_DEFAULT_WEIGHTS);
    brierFitted  = Math.round(bF * 1000) / 1000;
    brierDefault = Math.round(bD * 1000) / 1000;
    if (bF < bD - 0.005 && weightsSane(fitted)) { weights = fitted; mode = "FITTED"; }
    else { mode = "DEFAULTED"; }
  }

  function predict(aId: string, bId: string): PairwisePrediction {
    const directGames = state.games.filter(g =>
      (g.winner_id === aId && g.loser_id === bId) ||
      (g.winner_id === bId && g.loser_id === aId)
    ).length;
    const base = katzPredict(katz, aId, bId, directGames);
    const h2hGames = state.games.filter(g =>
      (g.winner_id === aId && g.loser_id === bId) ||
      (g.winner_id === bId && g.loser_id === aId)
    );
    const x = engineFeatures(aId, bId, eloMap, katz, katzStd, obt, h2hGames);
    const p = Math.max(0.02, Math.min(0.98, ensembleSig(ensembleDot(weights, x))));
    return { ...base, p_a_wins: p };
  }

  return {
    predict,
    ordinalBT: obt,
    status: {
      ensembleWeights: weights,
      ensembleMode: mode,
      looN: looEx.length,
      looBrierFitted: brierFitted,
      looBrierDefault: brierDefault,
      ordinalBTFitted: obt !== null,
      sigma: obt?.sigma ?? null,
    },
  };
}
