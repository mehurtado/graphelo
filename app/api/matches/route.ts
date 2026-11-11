import { NextRequest, NextResponse } from "next/server";
import { requireWriteAccess } from "@/lib/write-auth";
import { isObject, isStats, validateMatchInput } from "@/lib/match-input";
export const dynamic = "force-dynamic";
import { loadState, saveGame } from "@/lib/storage";
import type { Game } from "@/lib/graph-engine";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: NextRequest) {
  const denied = requireWriteAccess(req);
  if (denied) return denied;
  try {
    const body: unknown = await req.json().catch(() => null);
    if (!isObject(body)) {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }
    const invalid = validateMatchInput(body);
    if (invalid) return NextResponse.json({ error: invalid }, { status: 400 });
    const { winner_id, loser_id, winner_stats, loser_stats, score_winner, score_loser } = body;
    if (typeof winner_id !== "string" || typeof loser_id !== "string" || !winner_id || !loser_id) return NextResponse.json({ error: "winner_id and loser_id required" }, { status: 400 });
    if (winner_id === loser_id) return NextResponse.json({ error: "Players must be different" }, { status: 400 });

    const state = await loadState();
    if (!Object.hasOwn(state.players, winner_id)) return NextResponse.json({ error: `Unknown player: ${winner_id}` }, { status: 400 });
    if (!Object.hasOwn(state.players, loser_id)) return NextResponse.json({ error: `Unknown player: ${loser_id}` }, { status: 400 });

    const game: Game = {
      id: uuidv4(),
      timestamp: Date.now(),
      winner_id,
      loser_id,
      winner_stats: isStats(winner_stats) ? winner_stats : { kills: 0, deaths: 0 },
      loser_stats: isStats(loser_stats) ? loser_stats : { kills: 0, deaths: 0 },
      ...(score_winner !== undefined && score_loser !== undefined
        ? { score_winner: Number(score_winner), score_loser: Number(score_loser) }
        : {}),
    };
    await saveGame(game);
    state.games.push(game);
    return NextResponse.json(state, { status: 201 });
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
