import { NextResponse } from "next/server";
import { loadState } from "@/lib/storage";

export async function GET() {
  try {
    const state = await loadState();
    return NextResponse.json(state);
  } catch {
    return NextResponse.json({ error: "Failed to load state" }, { status: 500 });
  }
}
