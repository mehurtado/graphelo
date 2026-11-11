import { NextRequest, NextResponse } from "next/server";
import { requireWriteAccess } from "@/lib/write-auth";
export const dynamic = "force-dynamic";
import { loadState, savePlayer } from "@/lib/storage";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: NextRequest) {
  const denied = requireWriteAccess(req);
  if (denied) return denied;
  try {
    const body: unknown = await req.json().catch(() => null);
    if (!body || typeof body !== "object" || !("display_name" in body)) {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }
    const { display_name } = body;
    if (typeof display_name !== "string" || !display_name.trim()) return NextResponse.json({ error: "display_name required" }, { status: 400 });

    const state = await loadState();
    const exists = Object.values(state.players).find(
      p => p.display_name.toLowerCase() === display_name.trim().toLowerCase()
    );
    if (exists) return NextResponse.json({ error: "Player already exists" }, { status: 409 });

    const player = { id: uuidv4(), display_name: display_name.trim(), created_at: Date.now() };
    await savePlayer(player);
    state.players[player.id] = player;
    return NextResponse.json(state, { status: 201 });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[POST /api/players]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
