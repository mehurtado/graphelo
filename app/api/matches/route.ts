import { NextRequest, NextResponse } from "next/server";
export const dynamic = "force-dynamic";
import { loadState, appendGame } from "@/lib/storage";
import type { Game } from "@/lib/graph-engine";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: NextRequest) {
  try {
    const { winner_id, loser_id, winner_stats, loser_stats } = await req.json();
    if (!winner_id || !loser_id) return NextResponse.json({ error: "winner_id and loser_id required" }, { status: 400 });
    if (winner_id === loser_id) return NextResponse.json({ error: "Players must be different" }, { status: 400 });

    const state = await loadState();
    if (!state.players[winner_id]) return NextResponse.json({ error: `Unknown player: ${winner_id}` }, { status: 400 });
    if (!state.players[loser_id]) return NextResponse.json({ error: `Unknown player: ${loser_id}` }, { status: 400 });

    const game: Game = {
      id: uuidv4(),
      timestamp: Date.now(),
      winner_id,
      loser_id,
      winner_stats: winner_stats ?? { kills: 0, deaths: 0, assists: 0, headshots: 0 },
      loser_stats:  loser_stats  ?? { kills: 0, deaths: 0, assists: 0, headshots: 0 },
    };
    await appendGame(game);

    return NextResponse.json(
      { players: state.players, games: [...state.games, game] },
      { status: 201 },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[POST /api/matches]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function GET() {
  try {
    const state = await loadState();
    return NextResponse.json([...state.games].reverse());
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[GET /api/matches]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
