import { NextRequest, NextResponse } from "next/server";
export const dynamic = "force-dynamic";
import { deleteGame, loadState } from "@/lib/storage";

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
