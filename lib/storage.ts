import { Redis } from "@upstash/redis";
import type { GraphState } from "./graph-engine";

const redis = Redis.fromEnv();
const STATE_KEY = "graphelo:state";

export async function loadState(): Promise<GraphState> {
  const state = await redis.get<GraphState>(STATE_KEY);
  return state ?? { players: {}, games: [] };
}

export async function saveState(state: GraphState): Promise<void> {
  await redis.set(STATE_KEY, state);
}
