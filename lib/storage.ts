import { Redis } from "@upstash/redis";
import type { GraphState, Player, Game } from "./graph-engine";

const redis = Redis.fromEnv();

// Players: Redis Hash  (field = player id, value = Player JSON)
// Games:   Redis List  (each element = Game JSON, oldest first)
const PLAYERS_KEY = "graphelo:players";
const GAMES_KEY   = "graphelo:games";

export async function loadState(): Promise<GraphState> {
  const [players, games] = await Promise.all([
    redis.hgetall<Record<string, Player>>(PLAYERS_KEY),
    redis.lrange<Game>(GAMES_KEY, 0, -1),
  ]);
  return {
    players: players ?? {},
    games:   games   ?? [],
  };
}

// Atomic: hset on a single field never overwrites another player.
export async function appendPlayer(player: Player): Promise<void> {
  await redis.hset(PLAYERS_KEY, { [player.id]: player });
}

// Atomic: rpush appends without touching existing entries.
export async function appendGame(game: Game): Promise<void> {
  await redis.rpush(GAMES_KEY, game);
}
