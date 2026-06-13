import { NextRequest, NextResponse } from "next/server";
export const dynamic = "force-dynamic";
import { deleteGame, loadState, saveGame } from "@/lib/storage";

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    await deleteGame(id);
    const state = await loadState();
    return NextResponse.json(state);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[DELETE /api/matches/[id]]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const { winner_stats, loser_stats, score_winner, score_loser } = await req.json();

    const state = await loadState();
    const game = state.games.find(g => g.id === id);
    if (!game) return NextResponse.json({ error: "Game not found" }, { status: 404 });

    if (winner_stats) game.winner_stats = { kills: Number(winner_stats.kills) || 0, deaths: Number(winner_stats.deaths) || 0 };
    if (loser_stats)  game.loser_stats  = { kills: Number(loser_stats.kills)  || 0, deaths: Number(loser_stats.deaths)  || 0 };

    if (score_winner === null || score_loser === null) {
      delete game.score_winner;
      delete game.score_loser;
    } else if (score_winner !== undefined && score_loser !== undefined) {
      if (Number(score_winner) <= Number(score_loser)) {
        return NextResponse.json({ error: "Winner score must be greater than loser score" }, { status: 400 });
      }
      game.score_winner = Number(score_winner);
      game.score_loser = Number(score_loser);
    }

    await saveGame(game);
    const refreshed = await loadState();
    return NextResponse.json(refreshed);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[PATCH /api/matches/[id]]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
