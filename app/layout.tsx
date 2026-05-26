import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GraphELO // R6 Custom Rating",
  description: "Graph-based predictive rating system for Rainbow Six custom games",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
