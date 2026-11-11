# GraphELO

GraphELO tracks head-to-head Rainbow Six Siege custom games. Log results and optional round scores and kill/death counts to see player rankings, matchup forecasts, rating history, and model diagnostics.

The default ranking combines simulated round-robin title odds and average pairwise win probability. The matchup model blends Elo, graph centrality, ordinal Bradley–Terry estimates, and head-to-head history. The dashboard also exposes Elo as a separate ranking sort.

## Run locally

1. Install the locked dependencies with `npm ci`.
2. Connect a private Vercel Blob store and set `BLOB_READ_WRITE_TOKEN` in `.env.local`.
3. Set `GRAPHELO_WRITE_PASSWORD` in `.env.local` to a new, strong password. Set the same variable in the deployment environment when deploying. With no password configured, the dashboard is read-only.
4. Run `npm run dev` and open [localhost:3000](http://localhost:3000).

The dashboard prompts for the write password when adding, editing, or deleting data. Use HTTPS for deployed instances. Configure a separate Blob store for development if you want isolated match data.

Run `npm run lint` and `npm run build` before deploying.
