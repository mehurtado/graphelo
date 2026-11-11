import assert from "node:assert/strict";
import { test } from "node:test";
import * as engine from "../lib/graph-engine.ts";

const now = Date.UTC(2026, 10, 11);
const day = 86_400_000;
const players = Object.fromEntries(["a", "b", "c"].map(id => [
  id, { id, display_name: id.toUpperCase(), created_at: now - 30 * day },
]));
const games = [["a","b",7,3],["b","c",7,5],["a","c",7,2],["b","a",7,6],["a","b",7,4],["c","b",7,5]]
  .map(([winner_id, loser_id, score_winner, score_loser], index) => ({
    id: String(index), timestamp: now - (6 - index) * day,
    winner_id, loser_id, score_winner, score_loser,
    winner_stats: { kills: 7, deaths: 4 }, loser_stats: { kills: 4, deaths: 7 },
  }));
const state = { players, games };

function seededRandom() {
  let seed = 12345;
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

test("fixed match history preserves Elo and seeded tournament rankings", t => {
  t.mock.method(Date, "now", () => now);
  const model = engine.buildUnifiedEngine(state, now);
  const simulation = engine.simulateRoundRobin(state, model, seededRandom());

  assert.deepEqual(engine.computeElo(state), { a: 855, b: 768, c: 777 });
  assert.deepEqual(simulation.champ_counts, { a: 856, b: 43, c: 28 });
  assert.deepEqual(simulation.ranking.map(entry => entry.player_id), ["a", "b", "c"]);
});
