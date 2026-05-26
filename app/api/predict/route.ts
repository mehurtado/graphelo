import { NextRequest, NextResponse } from "next/server";
import { loadState } from "@/lib/storage";
import { simulateRoundRobin, predictPairwise, headToHead } from "@/lib/graph-engine";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const a = searchParams.get("a");
    const b = searchParams.get("b");
    const mode = searchParams.get("mode"); // "h2h" for head to head

    const state = await loadState();

    if (a && b) {
      if (!state.players[a] || !state.players[b])
        return NextResponse.json({ error: "Unknown player" }, { status: 400 });
      if (mode === "h2h") return NextResponse.json(headToHead(state, a, b));
      return NextResponse.json(predictPairwise(state, a, b));
    }

    // Full round robin ranking
    const ranking = simulateRoundRobin(state);
    return NextResponse.json(ranking);
  } catch {
    return NextResponse.json({ error: "Failed" }, { status: 500 });
  }
}
