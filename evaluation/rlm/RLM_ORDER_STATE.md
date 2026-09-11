# RLM over BM25-ordered chunks: state of the work

> **Read this file first.** It is updated at every milestone: code, pilot, launch, each
> finished stage, gate and report. Where two statements disagree, the newer one wins; the
> changelog at the bottom dates every update.
> **Last updated: 2026-09-11 ~06:20 IST.**
> - The server `ro_root` is up; it was restarted with the user's OK.
> - The old raw output in `$HOME` is archived on NAS.
> - The revision-3 pilot PASSED the strict gate (30/30). Its transcripts show two more
>   mechanical failures. Roots filter `c.text.lower()` with capitalised terms, which never
>   match. They then pass the empty list to `llm_query`, which reads nothing.
> - Revision 4 addresses both. Next: one more 3-row pilot, then the full LOFT launch.

## Goal

The graph-order campaign (`feature/graph-order`) read every chunk through `--mode
structured`. That mode is a fixed map→reduce: the sub-model reads every group, and one
root call answers from the notes. The root never decides what to read, so the pipeline
behaves like RAG. This work makes the **agentic RLM root (the REPL loop, `RLM.run`) the
orchestrator** instead. The root receives **every chunk of the document as a `chunks`
list, in a pre-computed order**, and is **told how that order was made**. It can then
read the most promising chunks first and answer with confidence once it has seen the
evidence.

| arm | flag | order of `chunks` | what the root is told |
|---|---|---|---|
| RLM + BM25 + graph (query-agnostic) | `--chunk-order bm25-graph` | The question-free organization (`organization.organize`). On LOFT this is the same cached artifact the structured campaign read. | It was organized before the question was known, and chunks that share distinctive words sit next to each other. Find a hit, then read it together with its neighbours. |
| RLM + BM25 (query-aware) | `--chunk-order bm25-query` | Every chunk, ranked by BM25 against the bare question (`ChunkIndex.rank_all`). Zero-score chunks come last, in source order. | It is sorted by relevance, and `chunks[0]` is the best match. Read the top of the list first, and FINAL with confidence once it answers the question. |

**Shared by both arms:**
- the same 2000-character, zero-overlap partition of the document (`organization.CONFIG`);
- the tools `context`, `chunks`, `llm_query` and the `FINAL*` family, with no `search()`;
- the base-RLM settings: `--max-steps 50`, 40 sub-calls, 900 s run timeout, 32000-character slice cap;
- chunk reads of at most 8192 tokens, priced exactly; an over-budget read is refused, never truncated.

**What differs:** the order, the paragraph describing it, and the worked session. The
prompts are rendered from `CHUNKS_HELP`, `CHUNK_ORDER_NOTES`, `SEARCH_TERMS_CHUNKS` and
`CHUNK_WORKED_SESSIONS` in `rlm.py`.

## Decisions, and why

**Made by the user on 2026-09-10:**
- **Two arms only, with no source-order control.** As a result, a gain over base RLM cannot be attributed to the ordering alone, and on qampari and quest it also carries the list-shape note. The clean pair is **`bm25-query` vs `bm25-graph`**.
- **The sorted list only, with no `search()`.** This keeps the graph arm query-agnostic in what it is given. Second hops go through keyword filters over `chunks`.
- **Chunks arrive as a REPL variable only**, with no preview in the first message. The worked example makes the first move.
- **RULER-128k: all 13 tasks × 55 rows**, after LOFT.

**Design choices, each resting on earlier evidence:**
- **Worked sessions demonstrate each arm's intended move, never `.find`.** The root copies demonstrations (commit ec2c8a2).
- **The BM25 query is the bare question.** LOFT's instruction wrapper dropped recall@2 on nq from 0.95 to 0.00. For RULER, the query is the text after the last `Question:`.
- **Chunks are a variable, not something returned by `search()`.** A frozen set behind `search()` trapped the root into 11 identical searches (commit a1ff98f).
- **qampari and quest get the bracketed-list FINAL note.** The structured arms have `listfmt`; without it, correct lists score 0.000. Legacy agentic prompts are unchanged.
- **One chunk read is at most 8192 tokens**, the same as one structured reader group. It is priced with the served model's tokenizer on the complete reader message, and the gate requires `reader_input_tokens == reader_served_prompt_tokens`.

**Harness changes in revision 3, applied to ordered-chunk runs only (`4e0df25`).** Every other
agentic configuration keeps its old behaviour. Both changes apply to both arms equally, so the
query-vs-graph pair stays clean. Against base RLM they are two more differences, on top of the
list note.
- **A code cell holding `Final Answer: X` is told to call `FINAL(...)`.** Before, it got "rewrite it multi-line", which cannot be done for one statement. On the r2 pilot, 3 of 15 graph rows resubmitted that line until the repetition breaker replaced their answers. Counted per row as `prose_answer_cells`.
- **`FINAL_NONE` needs 3 chunk reads (`CHUNK_READS_BEFORE_ABSTAIN`).** The legacy bar was one sub-call. Earlier, the root is told how many of the chunks it has read, and to ask for one fact at a time. On the r2 pilot, all 3 musique query rows abstained after a single read of `chunks[:5]`; one had the gold in `chunks[0]`, but the root had asked all three hops at once. The existing cap of 2 rejections still applies, so a root can always abstain eventually. Counted as `premature_abstentions`.

## Where everything is

| what | where |
|---|---|
| branch | `feature/rlm-ordered-chunks`, cut from `feature/graph-order` @ f8f7551. Pushed to **`nas`** only (fox `/mnt/nas/gautammahale/retrieval_checkout`). |
| Mac checkout | `linear/benchmark`. Tests run with `.venv-mac`; black and isort are in `.venv`. |
| compute host | **cat**, GPU 0 (RTX A6000, UUID `GPU-de398869-9f4d-2f4e-e0ab-738deda8383e`). cat has 32 cores. |
| server | tmux **`ro_root`**, started 2026-09-11 06:04:44 with the user's OK. It replaced `go_root`, whose engine had been killed. vLLM 0.8.5 on **:8002** on GPU-de398869 (CUDA index 0): `--max-model-len 65536`, util 0.5 (24.3 GB), cores 2–5. Weights are on cat-local `/tmp/gautammahale/hf_cache`; the log is at `/tmp/gautammahale/rlmorder/logs/root.log`. **Stop it when this campaign's runs finish**; that was the user's condition for the restart. |
| code checkout | cat `/mnt/home/gautammahale/rlmorder/checkout` (code only) |
| pilot scratch | cat `/tmp/gautammahale/rlmorder/{results_pilot_r2,results_pilot_r3,logs,report,hf_datasets}`; primary group `infolab`. The r2 gate output is `report/pilot_r2_gate.txt`. |
| old pilot evidence | NAS `/mnt/nas/gautammahale/archive/rlmorder/`. `results_pilot_r1` and `logs_r1`: revision 1, stopped. `results_pilot_r2`, `logs_r2` and `pilot_r2_gate.txt`: revision 2, gate failed. Never resume into them. |
| LOFT organization cache | `/mnt/nas/gautammahale/archive/graphorder/organization`, the structured campaign's own cache. It was archived off `$HOME` on 2026-09-11, and it is the lane script's default `ORG`. Every graph-arm row should report `organization_cache_hit: true`. |
| venv on cat | `/mnt/nas/gautammahale/venvs/rlm/bin/python` |
| HF cache | `HF_HOME=/mnt/nas/gautammahale/hf_cache`; pilot `HF_DATASETS_CACHE=/tmp/gautammahale/rlmorder/hf_datasets` is a copy of the five 128k caches. Offline lanes. |
| comparators | Structured arms: `/mnt/nas/gautammahale/archive/graphorder/results`, report in `archive/graphorder/report/`. Their configs still record the old `$HOME` organization dir, so pass the gate `--organization-dir`. Base RLM and vanilla: `/mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards`. |

## Status board

| stage | nq | hotpotqa | musique | qampari | quest |
|---|---|---|---|---|---|
| revision-3 code + tests (`4e0df25`) | 558 passed; black, isort, flake8 and mypy clean | | | | |
| revision-1 pilot | invalid: stopped | stopped | stopped | not started | not started |
| revision-2 pilot (3 rows × 2 arms) | done | gate FAIL (graph row 2) | gate FAIL (graph row 0) | gate FAIL (graph row 2) | done |
| revision-3 pilot (3 rows × 2 arms) | gate PASSED, 30/30 (06:12); archived | passed | passed | passed | passed |
| revision-4 pilot (3 rows × 2 arms) | todo | todo | todo | todo | todo |
| LOFT full (55 rows × 2 arms) | todo | todo | todo | todo | todo |
| RULER-128k (13 tasks × 55 × 2) | not started | | | | |

**The comparator to beat.** These are the structured graph-order results on the same 55
test rows. That campaign's gate passed on 2026-09-10.

| subset | metric | structured source | structured bm25-graph |
|---|---|---:|---:|
| nq | subspan_em | 0.709 | 0.764 |
| hotpotqa | subspan_em | 0.491 | 0.527 |
| musique | subspan_em | 0.109 | 0.255 |
| qampari | coverage | 0.599 | 0.479 |
| quest | coverage | 0.467 | 0.364 |

Those runs cost about 126–128k input tokens and 35–56 s per question: every chunk was read.

## How to resume (exact commands)

```bash
# 0. The server is tmux ro_root, and it must GENERATE (check below). If it is dead or wedged, kill its
#    session first, because vLLM refuses a port that is still listening. Then start it with the next command.
ssh cat.cse.iitb.ac.in 'tmux kill-session -t ro_root; sleep 5; ss -ltn | grep -c ":8002 "; nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader'
ssh cat.cse.iitb.ac.in 'tmux new-session -d -s ro_root "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/gautammahale/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/tmp/gautammahale/cache VLLM_CACHE_ROOT=/tmp/gautammahale/cache/vllm TRITON_CACHE_DIR=/tmp/gautammahale/cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/gautammahale/cache/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false UV_USE_IO_URING=0 taskset -c 2-5 /mnt/nas/gautammahale/benchmark/.venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name Qwen/Qwen3-4B-Instruct-2507 --port 8002 --max-model-len 65536 --gpu-memory-utilization 0.5 --enforce-eager -O0 > /tmp/gautammahale/rlmorder/logs/root.log 2>&1"'
# Once loaded, confirm CUDA index 0 is GPU-de398869-9f4d-2f4e-e0ab-738deda8383e: nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv
# The generation check. A wedged vLLM still answers /v1/models.
ssh cat.cse.iitb.ac.in 'curl -fsS --max-time 30 localhost:8002/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-4B-Instruct-2507\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}"'

# 1. Revision-3 pilot: ten lanes (arm × subset), the first 3 test rows, cores 6-15, under infolab.
#    Do not launch a duplicate active session.
ssh cat.cse.iitb.ac.in 'mkdir -p /tmp/gautammahale/rlmorder/results_pilot_r3; core=6; for arm in bm25-graph bm25-query; do for s in nq hotpotqa musique qampari quest; do tmux new-session -d -s "ro3_${arm#bm25-}_$s" "sg infolab -c \"LIMIT=3 HF_DATASETS_CACHE=/tmp/gautammahale/rlmorder/hf_datasets RESULTS=/tmp/gautammahale/rlmorder/results_pilot_r3 /mnt/home/gautammahale/rlmorder/checkout/evaluation/rlm/rlm_order_lane.sh $arm $core $s > /tmp/gautammahale/rlmorder/logs/pilot_r3_${arm}_$s.log 2>&1\""; core=$((core+1)); done; done; tmux ls'

# 2. Gate the complete revision-3 LOFT pilot against the first 3 structured test rows. Save the output.
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && PYTHONDONTWRITEBYTECODE=1 /mnt/nas/gautammahale/venvs/rlm/bin/python -m evaluation.rlm.rlm_order_gate /tmp/gautammahale/rlmorder/results_pilot_r3 --pilot --expect 3 --structured-results /mnt/nas/gautammahale/archive/graphorder/results --organization-dir /mnt/nas/gautammahale/archive/graphorder/organization | tee /tmp/gautammahale/rlmorder/report/pilot_r3_gate.txt'

# 3. Full LOFT: ten lanes (arm × subset), all 55 test rows, cores 6-15. Live output goes to cat-local scratch.
ssh cat.cse.iitb.ac.in 'T=/tmp/gautammahale/rlmorder; sg infolab -c "mkdir -p $T/results"; core=6; for arm in bm25-graph bm25-query; do for s in nq hotpotqa musique qampari quest; do tmux new-session -d -s "ro_${arm#bm25-}_$s" "sg infolab -c \"LIMIT=55 HF_DATASETS_CACHE=$T/hf_datasets RESULTS=$T/results /mnt/home/gautammahale/rlmorder/checkout/evaluation/rlm/rlm_order_lane.sh $arm $core $s > $T/logs/full_${arm}_$s.log 2>&1\""; core=$((core+1)); done; done; tmux ls'

# 4. Gate the full run (save the output), then write the report to local scratch. Reports never go in $HOME.
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && PYTHONDONTWRITEBYTECODE=1 /mnt/nas/gautammahale/venvs/rlm/bin/python -m evaluation.rlm.rlm_order_gate /tmp/gautammahale/rlmorder/results --expect 55 --structured-results /mnt/nas/gautammahale/archive/graphorder/results --organization-dir /mnt/nas/gautammahale/archive/graphorder/organization | tee /tmp/gautammahale/rlmorder/report/full_gate.txt'
ssh cat.cse.iitb.ac.in 'cd /mnt/home/gautammahale/rlmorder/checkout && HF_HOME=/mnt/nas/gautammahale/hf_cache PYTHONDONTWRITEBYTECODE=1 /mnt/nas/gautammahale/venvs/rlm/bin/python -m evaluation.rlm.rlm_order_report --results /tmp/gautammahale/rlmorder/results --out /tmp/gautammahale/rlmorder/report/full --structured-results /mnt/nas/gautammahale/archive/graphorder/results --guards /mnt/nas/gautammahale/benchmark/evaluation/results/loft128k_guards --repo .'

# 5. Archive every gated stage to NAS as infolab. /tmp is not durable. An empty verify (rsync -c -n) means identical.
ssh cat.cse.iitb.ac.in 'sg infolab -c "for d in results report logs; do rsync -a /tmp/gautammahale/rlmorder/\$d/ /mnt/nas/gautammahale/archive/rlmorder/\$d/ && rsync -a -c -n --itemize-changes /tmp/gautammahale/rlmorder/\$d/ /mnt/nas/gautammahale/archive/rlmorder/\$d/; done"'
```

The pilot runs ten short lanes (arm × subset) to test the concurrency planned for
full LOFT. Logs live under `/tmp/gautammahale/rlmorder/logs/pilot_r3_<arm>_<subset>.log`.
All 30 rows, strict gate, and transcript/list-shape inspection must pass before any
full launch. Never accept a pass on only the subsets that happen to exist.

**RULER-128k:**
1. Stage the dataset once, with network access on and outside `HF_HUB_OFFLINE`, into the NAS HF cache. Load each of the 13 splits of `lighteval/RULER-131072-Qwen-3`.
2. Launch the lanes with `DATASET=ruler128k`, passing the task names as the subsets. RULER has no split, so the lane script passes none.

## Traps
- **vLLM 0.8.5 parses `CUDA_VISIBLE_DEVICES` as an integer**, so a GPU UUID crashes startup. Resolve the UUID to an index; `serve_root.sh` does this.
- **`taskset` does not cover libuv's io_uring polling thread.** Set `UV_USE_IO_URING=0`; `serve_root.sh` does this.
- **A wedged vLLM still answers `/v1/models`.** Always preflight by generating; the lane script does.
- **Killing the PID that `nvidia-smi` shows kills only vLLM's engine core.** The API process keeps listening, answers `/v1/models`, and accepts chat requests that never run (2026-09-10 21:46:55). Look for a `<defunct>` child under the API PID (`ps --ppid <pid>`). To stop a server, kill its whole tmux session.
- **The ssh host is `cat.cse.iitb.ac.in`.** There is no `cat` alias on the Mac, and `ssh cat` stalls on DNS.
- **One example at n=55 is 0.018.** Do not report deltas smaller than that.
- **qampari and quest are scored on coverage, never on subspan_em.**
- **The guards runs have 110 rows: dev rows 0–9, then test.** Our test row i is guards position i+10. The report aligns them and checks the gold answers row by row.
- **Follow the lab storage policy (relayed by the user on 2026-09-11).**
  - `$HOME` (`/mnt/home`: NFS, backed up nightly) holds code, manuscripts and condensed output ONLY. Never raw data, fine-grained run logs, reports or diagrams.
  - Scratch belongs on `/mnt/cat/data/gautammahale`, which is NOT allocated yet; the advisor has to allocate it. Until it is, live raw output goes to cat-local `/tmp/gautammahale/rlmorder` (138 GiB free). That is not durable, so archive it after each stage.
  - Archival goes to `/mnt/nas/gautammahale` (720 GB free; the volume is 99% full). `/mnt/infonas/data` is not mounted on cat.
  - Write as primary group `infolab` (`sg infolab -c ...` or `exec newgrp infolab`). Keep envs and caches on local disk.
  - **The earlier violations were resolved on 2026-09-11, 06:06–06:11, with the user's OK.**
    - About 960 MB of raw output from `graphorder/`, `qagnostic/` and `rlmorder/` was rsynced to `/mnt/nas/gautammahale/archive/{graphorder,qagnostic,rlmorder}/` as `infolab`.
    - Every directory was checksum-verified (0 diffs), then deleted from `$HOME`. Only checkouts, scripts, docs and `qagnostic/ids` remain there.
    - The lane script's `ORG` default and the gate and report paths now point at the archive.
- **Never run `make style` on the Mac.** Its uv sync corrupts `.venv-mac`. Run `.venv/bin/python -m black/isort` directly.
- **Only doc-only commits may be pulled into a checkout whose lanes are running.**
- **Show loft128k numbers only** (standing user rule). Ask before showing any RULER-128k numbers.

## Next steps (ranked)
1. **Get the user's OK to restart the server on cat GPU 0.** Restart it with step 0 above, and check that it generates.
2. Launch the ten revision-3 pilot lanes (step 1), then gate all 30 rows (step 2). Inspect the transcripts:
   - no `Final Answer:` loops;
   - every abstention comes only after at least 3 reads;
   - the qampari and quest lists parse.
   A failure blocks the full campaign; keep failed rows as evidence.
3. Full LOFT: 55 rows × 5 subsets × 2 arms, with `LIMIT=55`, same lanes. Get the output location from the user first; cat-local `/tmp` is not durable. Then gate, report, and copy condensed evidence to NAS. The user's original request covers this run.
4. RULER-128k follows the LOFT gate, as planned.

## Changelog (newest first)
- **2026-09-11, 06:04–06:20 IST: server restarted, `$HOME` archived, and the r3 pilot PASSED.**
  - **The server.** The user OK'd a restart. `ro_root` has been up since 06:04:44 (ready 06:05:53) on GPU-de398869.
  - **Storage.** At 06:06–06:11, about 960 MB of old raw output was archived to NAS as `infolab`, checksum-verified, and deleted from `$HOME`.
  - **The r3 pilot** ran 06:08:17–06:11:58. The strict gate passed 30/30 rows, with no forced endings.
    - All 6 `Final Answer:` cells were corrected on the next step.
    - 9 early abstentions were rejected, and 3 of those rows went on to the right answer.
    - The query arm reads `chunks[:5]` first; the graph arm reads 8-chunk neighbourhoods. The lists parse.
    - The pilot is archived as `archive/rlmorder/{results_pilot_r3,logs_r3,pilot_r3_gate.txt}`.
  - **What the r3 transcripts show:**
    - Filters like `"Boston" in c.text.lower()` never match, because the term is capitalised.
    - `llm_query(q, [])` then reads nothing, the reader says "not mentioned", and the root abstains or writes a degenerate list.
    - One root also wrote `FINAL(Spain)` without quotes.
  - **Revision 4 therefore:**
    - adds a lower-case note to the shared search-terms bullet;
    - refuses an empty chunk list in chunk mode;
    - quotes the FINAL diagnosis.
- **2026-09-11 ~06:00 IST: the r2 pilot was gated (FAILED), revision 3 landed, and the server was found down.**
  - **r2 pilot.** All ten lanes printed DONE by 21:46:25, 30 rows in total. The gate failed on 3 graph rows ending `repetition_broken`: hotpotqa row 2, musique row 0 and qampari row 2. Each root read the right neighbourhood, then looped on a `Final Answer: X` code cell.
  - **What the r2 pilot shows about each arm.** The query arm now opens with `llm_query(q, chunks[:5])` and usually finishes in 2 steps, on about 2.9k root tokens. The graph arm finds its rows by keyword, then reads `chunks[i-2:i+6]`.
  - **Revision 3 (`4e0df25`)** addresses both failure modes; see Decisions. It was pushed to `nas` and pulled on cat.
  - **The server.** The engine was killed at 21:46:55 from an interactive login (`kill 15895`, then `kill -9 15895`), which freed GPU 0. The API on :8002 is wedged. It was not restarted, pending the user's OK.
- **2026-09-10 21:33:58 IST: replacement pilot launched.** Code `cebce6c` is synchronized across Mac, NAS and cat. Ten lanes, first 3 test rows for each of five LOFT subsets and two arms, local scratch `/tmp/gautammahale/rlmorder/results_pilot_r2`, group `infolab`, cores 6–15. The 243 MiB prepared 128k dataset cache was copied locally; tokenizer remains in the configured NAS HF cache. Original pilot outputs are retained separately.
- **2026-09-10 ~21:30 IST: revision-2 prompt and gate fixes.** Stopped both revision-1 pilot process groups at 21:23, preserving their outputs and the go_root server. Removed ranked Hit-print demonstrations, made the first query move a reader call, clarified representations, and added graph retry guidance. Bumped chunk_order_revision to 2. Gate now requires every task/arm, completed valid rows with reader calls, exact positive token counts, no limit/server notices, comparator IDs/question/gold/document identity, and cached graph-order identity. Added pilot prefix matching and regression tests. Lane failures propagate and new outputs require an explicit scratch path and infolab group. Full evaluation tests: 553 passed, 1 skipped.
- **2026-09-10 ~20:55 IST: code, tests and this file landed.**
  - Added the `--chunk-order {bm25-graph,bm25-query}` and `--chunk-call-tokens` flags, and `ChunkIndex.rank_all`.
  - `run()` gained `ordered_chunks`, and priced order-preserving chunk reads via `OrderedPayload`.
  - Added the `ruler128k` dataset.
  - Added `rlm_order_lane.sh`, `rlm_order_gate.py` and `rlm_order_report.py`.
  - Added `tests/evaluation/test_rlm_chunk_order.py` (58 tests). The full suite is 532 passed.
- **2026-09-10 20:25 IST: the graph-order campaign was gated (PASSED, 10/10 runs, 0 errors) and reported** to `graphorder/report/`.
