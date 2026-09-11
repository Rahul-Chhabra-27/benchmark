# Query-agnostic campaign: host scripts (2026-09-09)

These are the lane scripts the query-agnostic campaign ran on its host, kept exactly as
they ran. The campaign itself is described in [`QAGNOSTIC_HANDOFF.md`](../QAGNOSTIC_HANDOFF.md).
Its host checkouts (`/mnt/home/gautammahale/qagnostic/checkout*`) were synced working
trees. They were checked on 2026-09-11 and are identical to `feature/query-agnostic`,
except for one older `run_benchmark.py` snapshot that holds nothing the branch lacks.

| script | what it did |
|---|---|
| `lane.sh <subset> <stage>` | One lane: one subset and one `structured_grid.sh` stage, against one vLLM on :8001. |
| `gen_chunks.sh` | Wrote the query-free chunk sets (`query_free_chunks.py`) and the holdout ids (test rows 55–99). |
| `notes_lane.sh <subset> [window] [overlap]` | Built the query-free notes index (`window_notes.py`), once per subset. |
| `autoshutdown.py` | Retired only the captured elk vLLM server once the campaign had exited. |

The paths in these scripts are the campaign's own, and they put raw output in `$HOME`.
That predates the lab storage policy. The output itself (results, logs, chunk sets and
notes) is archived at `/mnt/nas/gautammahale/archive/qagnostic/`, and `ids/` stayed in
`$HOME`. Before reusing any script, point `RESULTS`, `QCHUNKS` and `--out-dir` at scratch
or at the archive.
