import { put, list } from "@vercel/blob";
import type { GraphState } from "./graph-engine";

const BLOB_PATHNAME = "graphelo/state-v2.json";

// Throws on failure so API routes can log and surface the real error.
export async function loadState(): Promise<GraphState> {
  const { blobs } = await list({ prefix: BLOB_PATHNAME, limit: 1 });
  if (blobs.length === 0) return { players: {}, games: [] };

  // Private blobs require the token in the Authorization header.
  const res = await fetch(blobs[0].url, {
    cache: "no-store",
    headers: { Authorization: `Bearer ${process.env.BLOB_READ_WRITE_TOKEN}` },
  });
  if (!res.ok) throw new Error(`Blob read failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function saveState(state: GraphState): Promise<void> {
  await put(BLOB_PATHNAME, JSON.stringify(state), {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: "application/json",
  });
}
