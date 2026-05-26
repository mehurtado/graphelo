import { put, list } from "@vercel/blob";
import type { GraphState } from "./graph-engine";

const BLOB_PATHNAME = "graphelo/state-v2.json";

// Throws on failure so API routes can surface the real error.
export async function loadState(): Promise<GraphState> {
  const { blobs } = await list({ prefix: BLOB_PATHNAME, limit: 1 });
  if (blobs.length === 0) return { players: {}, games: [] };

  // Append timestamp to bust Vercel's CDN cache — otherwise stale content is served
  // after a write until the CDN edge TTL expires.
  const url = new URL(blobs[0].url);
  url.searchParams.set("_t", Date.now().toString());

  const res = await fetch(url.toString(), {
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
