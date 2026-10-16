# scripts — everything you run from a terminal

Run every script from the repository root: `bash scripts/rlm/run_bxf.sh`.
SLURM batch jobs (`sbatch …`) are in [`../slurm/`](../slurm/).

## `rlm/` — the RLM experiments

| script | what it does |
|---|---|
| `serve_root.sh` | start the root vLLM server on one GPU (in tmux) |
| `root_server.sh` | helpers the launchers source: reuse a running server or start one, stop it after |
| `run_bxf.sh` | one command for a B × F grid of the ordered-chunk arms: start server, run every cell, print the table |
| `run_picked.sh` | one command for `compression_picker`: pick F per question, run the groups, score |
| `rlm_order_lane.sh` | one lane (one arm, one CPU core) of the ordered-chunk campaign; `run_bxf.sh` and `run_picked.sh` call it |
| `rlm_order_postrun.sh` | after a stage's lanes finish: gate the results, stop the server, archive |
| `structured_grid.sh` | the chunk-geometry campaign for the structured (map → reduce) arm |
| `download_data.sh` | on the login node: fill the shared Hugging Face cache before GPU jobs run offline |

Launchers tied to one finished campaign, kept so it can be re-run as it was:

| script | campaign |
|---|---|
| `campaigns/loft128k_vanilla_vs_rlm/run_infolab.sh`, `…/run_cells.sh` | LOFT-128k vanilla vs RLM on the infolab hosts |
| `campaigns/fixed_chunk_grid/run_grid.sh` | fixed chunk × KV-retention grid |
| `campaigns/query_agnostic_chunks/lane.sh`, `…/notes_lane.sh` | query-agnostic chunk selection (2026-09) |

The clean RLM needs no script: `python -m evaluation.rlm.minimal_rlm run evaluation/rlm/minimal_rlm/configs/<config>.yaml`.

## `official_rlm/` — the official recursive-llm library (from apd10-dev)

| script | what it does |
|---|---|
| `run_qwen35_loft128k_n20.sh` | recursive-llm on LOFT-128k, 20 test rows × 5 tasks, Qwen3.5 4B / 9B / 27B in turn |
| `run_qwen35_loft128k_n20_split_parallel.sh` | the same against three running servers, all in parallel |
| `run_gemini_loft128k_n20.sh` | the same with Gemini 3.6 Flash |
| `run_gemini_35_flash_lite_loft128k_n20.sh` | the same with Gemini 3.5 Flash Lite |
| `stop_qwen35_campaign_servers.sh` | after a campaign: stop only its three Qwen3.5 servers |

These default to `REPO=/data/aditya_desai/agentic-longcontext`; set `REPO` to your checkout.
