# Research agent operating rules

- Use the configured research store and the library, CLI or MCP workflow for experiments.
- Keep the snapshot's reserved holdout interval fixed throughout development.
- Request the holdout only after reviewing the gate evidence, with explicit acknowledgement.
- Report gate verdicts and their reasons exactly as returned.
- Keep ledger and run artifacts under the managed workflow. Do not edit them manually.
