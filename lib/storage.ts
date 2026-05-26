import { put } from "@vercel/blob";
import type { GraphState } from "./graph-engine";

const BLOB_PATHNAME = "graphelo/state-v2.json";

export async function loadState(): Promise<GraphState> {
  try {
    const url = `https://blob.vercel-storage.com/${BLOB_PATHNAME}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("not found");
    return await res.json();
  } catch {
    return { players: {}, games: [] };
  }
}

export async function saveState(state: GraphState): Promise<void> {
  await put(BLOB_PATHNAME, JSON.stringify(state), {
    access: "public",
    addRandomSuffix: false,
    contentType: "application/json",
  });
}
