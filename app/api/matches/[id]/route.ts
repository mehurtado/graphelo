import { NextRequest, NextResponse } from "next/server";
import { requireWriteAccess } from "@/lib/write-auth";
import { isObject, isStats, validateMatchInput } from "@/lib/match-input";
export const dynamic = "force-dynamic";
import { deleteGame, loadState, saveGame } from "@/lib/storage";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const denied = requireWriteAccess(req);
  if (denied) return denied;
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
  const denied = requireWriteAccess(req);
  if (denied) return denied;
  try {
    const { id } = await params;
    const body: unknown = await req.json().catch(() => null);
    if (!isObject(body)) {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }
    const invalid = validateMatchInput(body, true);
    if (invalid) return NextResponse.json({ error: invalid }, { status: 400 });
    const { winner_stats, loser_stats, score_winner, score_loser } = body;

    const state = await loadState();
    const game = state.games.find(g => g.id === id);
    if (!game) return NextResponse.json({ error: "Game not found" }, { status: 404 });

    if (isStats(winner_stats)) game.winner_stats = { kills: winner_stats.kills, deaths: winner_stats.deaths };
    if (isStats(loser_stats)) game.loser_stats = { kills: loser_stats.kills, deaths: loser_stats.deaths };

    if (score_winner === null || score_loser === null) {
      delete game.score_winner;
      delete game.score_loser;
    } else if (score_winner !== undefined && score_loser !== undefined) {
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
