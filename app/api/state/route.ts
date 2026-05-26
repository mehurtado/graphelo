import { NextResponse } from "next/server";
import { loadState } from "@/lib/storage";

export async function GET() {
  try {
    const state = await loadState();
    return NextResponse.json(state);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[GET /api/state]", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
