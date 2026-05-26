import { NextRequest, NextResponse } from "next/server";
import { loadState, saveState } from "@/lib/storage";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { winner_id, loser_id, winner_stats, loser_stats } = body;

    if (!winner_id || !loser_id) return NextResponse.json({ error: "winner_id and loser_id required" }, { status: 400 });
    if (winner_id === loser_id) return NextResponse.json({ error: "Players must be different" }, { status: 400 });

    const state = await loadState();
    if (!state.players[winner_id]) return NextResponse.json({ error: `Unknown player: ${winner_id}` }, { status: 400 });
    if (!state.players[loser_id]) return NextResponse.json({ error: `Unknown player: ${loser_id}` }, { status: 400 });

    const game = {
      id: uuidv4(),
      timestamp: Date.now(),
      winner_id,
      loser_id,
      winner_stats: winner_stats ?? { kills: 0, deaths: 0, assists: 0, headshots: 0 },
      loser_stats: loser_stats ?? { kills: 0, deaths: 0, assists: 0, headshots: 0 },
    };

    state.games.push(game);
    await saveState(state);
    return NextResponse.json(game, { status: 201 });
  } catch {
    return NextResponse.json({ error: "Failed to log game" }, { status: 500 });
  }
}

export async function GET() {
  try {
    const state = await loadState();
    return NextResponse.json([...state.games].reverse());
  } catch {
    return NextResponse.json({ error: "Failed" }, { status: 500 });
  }
}
