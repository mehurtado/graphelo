import { put, list, del } from "@vercel/blob";
import type { GraphState, Player, Game } from "./graph-engine";

const PLAYERS_PREFIX = "graphelo/players/";
const GAMES_PREFIX   = "graphelo/games/";
const LEGACY_BLOB    = "graphelo/state-v2.json";
const LEGACY_PLAYERS = "graphelo/players.json";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    headers: { Authorization: `Bearer ${process.env.BLOB_READ_WRITE_TOKEN}` },
  });
  if (!res.ok) throw new Error(`Blob read failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function loadState(): Promise<GraphState> {
  const [{ blobs: pBlobs }, { blobs: gBlobs }] = await Promise.all([
    list({ prefix: PLAYERS_PREFIX, limit: 500 }),
    list({ prefix: GAMES_PREFIX,   limit: 1000 }),
  ]);

  let players: Record<string, Player> = {};

  if (pBlobs.length > 0) {
    const loaded = await Promise.all(pBlobs.map(b => fetchJson<Player>(b.url)));
    for (const p of loaded) players[p.id] = p;
  } else {
    // One-time migration: old single players.json blob
    const { blobs: legacyP } = await list({ prefix: LEGACY_PLAYERS, limit: 1 });
    if (legacyP.length > 0) {
      const old = await fetchJson<Record<string, Player>>(legacyP[0].url);
      await Promise.all(Object.values(old).map(p => savePlayer(p)));
      try { await del(legacyP[0].url); } catch { /* non-critical */ }
      players = old;
    } else {
      // Older migration: full state-v2.json blob
      const { blobs: legacy } = await list({ prefix: LEGACY_BLOB, limit: 1 });
      if (legacy.length > 0) {
        const old = await fetchJson<GraphState>(legacy[0].url);
        await Promise.all([
          ...Object.values(old.players).map(p => savePlayer(p)),
          ...old.games.map(g => saveGame(g)),
        ]);
        try { await del(legacy[0].url); } catch { /* non-critical */ }
        players = old.players;
      }
    }
  }

  const games: Game[] = gBlobs.length > 0
    ? await Promise.all(gBlobs.map(b => fetchJson<Game>(b.url)))
    : [];
  games.sort((a, b) => a.timestamp - b.timestamp);

  return { players, games };
}

export async function savePlayer(player: Player): Promise<void> {
  await put(`${PLAYERS_PREFIX}${player.id}.json`, JSON.stringify(player), {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: "application/json",
  });
}

export async function deletePlayer(id: string): Promise<void> {
  const { blobs } = await list({ prefix: `${PLAYERS_PREFIX}${id}.json`, limit: 1 });
  if (blobs.length > 0) await del(blobs[0].url);
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
