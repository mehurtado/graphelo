import { put, list } from "@vercel/blob";
import type { GraphState } from "./graph-engine";

const BLOB_PATHNAME = "graphelo/state-v2.json";

export async function loadState(): Promise<GraphState> {
  try {
    const { blobs } = await list({ prefix: BLOB_PATHNAME, limit: 1 });
    if (blobs.length === 0) return { players: {}, games: [] };
    const res = await fetch(blobs[0].downloadUrl, { cache: "no-store" });
    if (!res.ok) return { players: {}, games: [] };
    return await res.json();
  } catch {
    return { players: {}, games: [] };
  }
}

export async function saveState(state: GraphState): Promise<void> {
  await put(BLOB_PATHNAME, JSON.stringify(state), {
    access: "private",
    addRandomSuffix: false,
    contentType: "application/json",
  });
}
