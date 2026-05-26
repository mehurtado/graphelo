import { NextRequest, NextResponse } from "next/server";
import { loadState, saveState } from "@/lib/storage";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: NextRequest) {
  try {
    const { display_name } = await req.json();
    if (!display_name?.trim()) return NextResponse.json({ error: "display_name required" }, { status: 400 });

    const state = await loadState();
    const exists = Object.values(state.players).find(
      p => p.display_name.toLowerCase() === display_name.trim().toLowerCase()
    );
    if (exists) return NextResponse.json({ error: "Player already exists" }, { status: 409 });

    const id = uuidv4();
    state.players[id] = { id, display_name: display_name.trim(), created_at: Date.now() };
    await saveState(state);
    return NextResponse.json(state.players[id], { status: 201 });
  } catch {
    return NextResponse.json({ error: "Failed to add player" }, { status: 500 });
  }
}
