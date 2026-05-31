import { put, list, del } from "@vercel/blob";
import type { GraphState, Player, Game } from "./graph-engine";

const PLAYERS_BLOB  = "graphelo/players.json";
const GAMES_PREFIX  = "graphelo/games/";
const LEGACY_BLOB   = "graphelo/state-v2.json";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    headers: { Authorization: `Bearer ${process.env.BLOB_READ_WRITE_TOKEN}` },
  });
  if (!res.ok) throw new Error(`Blob read failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function loadState(): Promise<GraphState> {
  const { blobs: pBlobs } = await list({ prefix: PLAYERS_BLOB, limit: 1 });

  let players: Record<string, Player>;
  if (pBlobs.length > 0) {
    players = await fetchJson(pBlobs[0].url);
  } else {
    // One-time migration from legacy single-blob format — only runs when no players blob exists
    const { blobs: legacy } = await list({ prefix: LEGACY_BLOB, limit: 1 });
    if (legacy.length > 0) {
      const old = await fetchJson<GraphState>(legacy[0].url);
      await savePlayers(old.players);
      await Promise.all(old.games.map(g => saveGame(g)));
      try { await del(legacy[0].url); } catch { /* non-critical */ }
      players = old.players;
    } else {
      players = {};
    }
  }

  const { blobs: gBlobs } = await list({ prefix: GAMES_PREFIX, limit: 1000 });
  const games: Game[] = gBlobs.length > 0
    ? await Promise.all(gBlobs.map(b => fetchJson<Game>(b.url)))
    : [];
  games.sort((a, b) => a.timestamp - b.timestamp);

  return { players, games };
}

export async function savePlayers(players: Record<string, Player>): Promise<void> {
  await put(PLAYERS_BLOB, JSON.stringify(players), {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: "application/json",
  });
}

export async function saveGame(game: Game): Promise<void> {
  await put(`${GAMES_PREFIX}${game.id}.json`, JSON.stringify(game), {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: "application/json",
  });
}

export async function deleteGame(id: string): Promise<void> {
  const { blobs } = await list({ prefix: `${GAMES_PREFIX}${id}.json`, limit: 1 });
  if (blobs.length > 0) await del(blobs[0].url);
}
