export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function isStats(value: unknown): value is { kills: number; deaths: number } {
  return isObject(value) && isCount(value.kills) && isCount(value.deaths);
}

export function validateMatchInput(body: Record<string, unknown>, editing = false): string | null {
  const { winner_stats, loser_stats, score_winner, score_loser } = body;
  if ([winner_stats, loser_stats].some(stats => stats !== undefined && !isStats(stats))) {
    return "Kills and deaths must be nonnegative integers";
  }
  if (score_winner === undefined && score_loser === undefined) return null;
  if (editing && score_winner === null && score_loser === null) return null;
  if (!isCount(score_winner) || !isCount(score_loser) || score_winner <= score_loser) {
    return "Provide both scores as nonnegative integers with winner score greater than loser score";
  }
  return null;
}
