import { createHash, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

export function requireWriteAccess(req: NextRequest) {
  const password = process.env.GRAPHELO_WRITE_PASSWORD;
  if (!password) {
    return NextResponse.json({ error: "Writes are disabled" }, { status: 503 });
  }

  const digest = (value: string) => createHash("sha256").update(value).digest();
  const provided = req.headers.get("X-GraphELO-Password") ?? "";
  if (!timingSafeEqual(digest(provided), digest(encodeURIComponent(password)))) {
    return NextResponse.json({ error: "Incorrect write password" }, { status: 401 });
  }
  return null;
}
