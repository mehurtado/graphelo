import { NextRequest, NextResponse } from "next/server";
export const dynamic = "force-dynamic";
import { loadState, savePlayers, deleteGame } from "@/lib/storage";

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const state = await loadState();

    if (!state.players[id]) return NextResponse.json({ error: "Unknown player" }, { status: 404 });

    const playerGames = state.games.filter(g => g.winner_id === id || g.loser_id === id);
    await Promise.all(playerGames.map(g => deleteGame(g.id)));

    delete state.players[id];
    await savePlayers(state.players);

    state.games = state.games.filter(g => g.winner_id !== id && g.loser_id !== id);
    return NextResponse.json(state);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[DELETE /api/players/[id]]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
