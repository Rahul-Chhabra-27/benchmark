# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run LOFT-128k with the recursive-llm library against a local vLLM server.

The long document stays in the REPL as `context`. The model explores it with
Python and may call `rlm_query` / `recursive_llm` (aliases; the latter is the
paper name) on selected slices. At the default max_depth=1 that call is one
plain LM completion.

The process always runs in ``baselines/official_rlm/recursive-llm/.venv`` (the lockfile
environment from that project). Invoking this file from another interpreter
re-execs into that venv. Do not install recursive-llm into the main repo venv.

Usage:
  ~/agentic-longcontext/baselines/recursive-llm/.venv/bin/python \\
      baselines/official_rlm/run_benchmark.py --endpoint http://localhost:8000/v1 --limit 1
  python baselines/official_rlm/run_benchmark.py --endpoint http://127.0.0.1:8000 \\
      --model Qwen/Qwen3-4B-Instruct-2507 --task nq_128k --split test

Create the original environment with:
  cd ~/agentic-longcontext/baselines/recursive-llm && uv sync

Provider keys (GEMINI_API_KEY, OPENAI_API_KEY, ...) are read from the
environment, or from a `.env` beside this repo or the recursive-llm checkout.

With ``--endpoint``, Qwen3.5 thinking traces that vLLM concatenates into
``message.content`` (``<think>...</think>`` or a trailing ``</think>``) are
stripped before recursive-llm's FINAL/REPL parsers run. Gemini API calls are
left unchanged; they already put thoughts in ``reasoning_content``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urlparse, urlunparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from evaluation.benchmarks.registry import LOFT_RAG_DATASETS, SCORER_REGISTRY  # noqa: E402
from evaluation.benchmarks.results import score_prediction_frame  # noqa: E402
from evaluation.results_layout import RLM_RESULTS_DIR  # noqa: E402
from evaluation.rlm.shared.datasets import load_examples  # noqa: E402

REEXEC_FLAG = "BASELINES_RECURSIVE_LLM_ENV"

LOFT_128K_TASKS = tuple(f"{name}_128k" for name in LOFT_RAG_DATASETS)
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
KNOWN_PROVIDERS = (
    "openai",
    "hosted_vllm",
    "vllm",
    "ollama",
    "openai_compatible",
    "gemini",
    "vertex_ai",
    "vertex_ai_beta",
    "anthropic",
    "groq",
    "deepseek",
    "mistral",
    "azure",
    "bedrock",
    "cohere",
    "openrouter",
)

_THINK_BLOCK = re.compile(r"<think\b[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


def split_vllm_reasoning(content: str) -> tuple[str, str]:
    """Split a vLLM chat string into (reasoning, protocol content).

    Gemini already puts thoughts in a side channel. Qwen3.5 on vLLM without a
    reasoning parser writes them into ``message.content`` and closes with
    ``</think>``. recursive-llm's FINAL matcher and REPL require the protocol
    turn to be the whole remaining string.
    """
    if not content:
        return "", ""
    blocks = _THINK_BLOCK.findall(content)
    if blocks:
        reasoning = "\n".join(part.strip() for part in blocks if part.strip())
        protocol = _THINK_BLOCK.sub("", content).strip()
        return reasoning, protocol
    parts = _THINK_CLOSE.split(content, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", content.strip()


def _response_field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def protocol_text_from_completion(response: Any, content: str) -> str:
    """Prefer provider-split reasoning; otherwise peel think tags out of content."""
    choices = _response_field(response, "choices")
    message = None
    if isinstance(choices, (list, tuple)) and choices:
        message = _response_field(choices[0], "message")
    provider_reasoning = _response_field(message, "reasoning_content") or _response_field(message, "reasoning")
    if isinstance(provider_reasoning, dict):
        provider_reasoning = provider_reasoning.get("content") or provider_reasoning.get("text")
    if isinstance(provider_reasoning, str) and provider_reasoning.strip():
        _, protocol = split_vllm_reasoning(content)
        return protocol if protocol else content.strip()
    _, protocol = split_vllm_reasoning(content)
    return protocol


def install_vllm_reasoning_split() -> None:
    """Make every RLM in this process (including child RLMs) drop vLLM think traces."""
    from rlm.core import RLM

    current = RLM._response_text_info
    if getattr(current, "_vllm_reasoning_split", False):
        return

    def _response_text_info(response: Any) -> tuple[str, bool]:
        text, normalized_none = current(response)
        protocol = protocol_text_from_completion(response, text)
        return protocol, normalized_none

    _response_text_info._vllm_reasoning_split = True  # type: ignore[attr-defined]
    RLM._response_text_info = staticmethod(_response_text_info)  # type: ignore[method-assign]


def recursive_llm_root() -> Path:
    """The local recursive-llm checkout that owns the original uv environment."""
    override = os.environ.get("RECURSIVE_LLM_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    home = Path.home() / "agentic-longcontext" / "baselines" / "official_rlm" / "recursive-llm"
    repo = REPO_ROOT / "baselines" / "official_rlm" / "recursive-llm"
    for candidate in (home, repo):
        if (candidate / "src" / "rlm").is_dir():
            return candidate.resolve()
    return repo.resolve()


def recursive_llm_python() -> Path:
    return recursive_llm_root() / ".venv" / "bin" / "python"


def recursive_llm_src() -> Path:
    return recursive_llm_root() / "src"


def in_recursive_llm_env() -> bool:
    python = recursive_llm_python()
    return python.exists() and Path(sys.executable).resolve() == python.resolve()


def reexec_into_recursive_llm_env(argv: Optional[Sequence[str]] = None) -> None:
    """Replace this process with recursive-llm's .venv interpreter if needed."""
    if os.environ.get(REEXEC_FLAG) == "1" or in_recursive_llm_env():
        os.environ[REEXEC_FLAG] = "1"
        return
    python = recursive_llm_python()
    if not python.is_file():
        root = recursive_llm_root()
        raise SystemExit(f"recursive-llm venv not found at {python}. Create it with: cd {root} && uv sync")
    env = os.environ.copy()
    env[REEXEC_FLAG] = "1"
    env["VIRTUAL_ENV"] = str(recursive_llm_root() / ".venv")
    pythonpath = [str(REPO_ROOT), str(recursive_llm_src())]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    args = [str(python), str(Path(__file__).resolve())]
    args.extend(list(argv) if argv is not None else sys.argv[1:])
    os.execve(str(python), args, env)


def ensure_recursive_llm_on_path() -> Path:
    """Prefer the local recursive-llm source over any other installed copy."""
    src = str(recursive_llm_src())
    if src not in sys.path:
        sys.path.insert(0, src)
    return recursive_llm_src()


def load_env_files() -> list[Path]:
    """Read provider keys from .env, so they never have to be typed on a command line.

    Matches what recursive-llm's own benchmark scripts do. Existing environment
    variables win, so an explicit `export` still overrides the file.
    """
    from dotenv import load_dotenv

    loaded = []
    for candidate in (recursive_llm_root() / ".env", REPO_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            loaded.append(candidate)
    return loaded


def normalize_endpoint(endpoint: str) -> str:
    """Accept a host, `:port`, or OpenAI-style `/v1` URL and return an api_base."""
    raw = endpoint.strip()
    if not raw:
        raise ValueError("endpoint must be a non-empty URL")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid vLLM endpoint: {endpoint!r}")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif path in ("/v1/chat/completions", "/chat/completions"):
        path = "/v1"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def litellm_model_id(model: str) -> str:
    """Prefix a served Hugging Face id so LiteLLM treats it as OpenAI-compatible."""
    name = model.strip()
    if not name:
        raise ValueError("model must be a non-empty served model id")
    provider = name.split("/", 1)[0]
    if provider in KNOWN_PROVIDERS:
        return name
    return f"openai/{name}"


def loft_128k_tasks(task: Optional[str]) -> tuple[str, ...]:
    """Resolve `--task` against the LOFT RAG 128k subsets."""
    if task is None or task in ("loft_128k", "all"):
        return LOFT_128K_TASKS
    if task in LOFT_128K_TASKS:
        return (task,)
    raise ValueError(f"unknown LOFT-128k task {task!r}; choose from {LOFT_128K_TASKS} or 'all'")


def model_slug(model: str) -> str:
    return litellm_model_id(model).replace("openai/", "").replace("/", "-")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI-compatible server URL (e.g. http://localhost:8000/v1 for vLLM). Omit for direct API providers.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model identifier (e.g. Qwen/Qwen3-4B-Instruct-2507, or gemini/gemini-2.5-flash for Gemini API)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (defaults to GEMINI_API_KEY, OPENAI_API_KEY, etc. from env, or 'EMPTY' if endpoint is set)",
    )
    parser.add_argument(
        "--task",
        default="loft_128k",
        help="one LOFT RAG 128k subset (nq_128k, ...) or loft_128k/all for every subset",
    )
    parser.add_argument("--split", default="test", choices=["all", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--out", default=str(Path(RLM_RESULTS_DIR) / "recursive-llm"))
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="recursive-llm max_depth (1 = recursive_llm is a leaf LM call)",
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--repl-timeout", type=float, default=60.0)
    parser.add_argument("--max-output-chars", type=int, default=2000)
    parser.add_argument("--max-elapsed-seconds", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0, help="per-HTTP LiteLLM timeout in seconds")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Qwen3.5 thinking ON (chat_template_kwargs.enable_thinking=True); default",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="disable Qwen3.5 thinking mode",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument(
        "--no-answer-prefix",
        action="store_true",
        help="do not append the LOFT answer_prefix (e.g. 'Final Answer: ') to the query",
    )
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def build_query(example: dict, include_answer_prefix: bool) -> str:
    question = example["question"]
    prefix = example.get("answer_prefix") or ""
    if include_answer_prefix and prefix.strip():
        return f"{question}\n{prefix}"
    return question


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"!! skipping a torn checkpoint line in {path}; that example will rerun")
            continue
        if not record.get("error"):
            done.add(record["id"])
    return done


def trajectory_as_dicts(result: Any) -> list[dict]:
    events = getattr(result, "trajectory", ()) or ()
    out = []
    for event in events:
        if hasattr(event, "to_dict"):
            out.append(event.to_dict())
        elif isinstance(event, dict):
            out.append(event)
    return out


def record_from_result(example: dict, query: str, result: Any, latency_s: float) -> dict:
    stats = getattr(result, "stats", None) or {}
    succeeded = bool(getattr(result, "succeeded", True))
    answer = getattr(result, "answer", None)
    record = {
        "id": example["id"],
        "dataset": "loft",
        "task": example.get("task") or "",
        "mode": "recursive-llm",
        "answers": example["answers"],
        "answer": example.get("scoring", {}).get("answer", example["answers"]),
        "answer_prefix": example.get("answer_prefix") or "",
        "query": query,
        "pred": answer,
        "predicted_answer": answer if answer is not None else "",
        "finished": succeeded,
        "end_reason": "completed" if succeeded else getattr(result, "error_type", "failed"),
        "steps": stats.get("iterations") or stats.get("total_iterations") or 0,
        "sub_calls": stats.get("recursive_calls") or stats.get("leaf_calls") or 0,
        "tokens": stats.get("total_tokens") or 0,
        "prompt_tokens": stats.get("prompt_tokens") or 0,
        "completion_tokens": stats.get("completion_tokens") or 0,
        "latency_s": round(latency_s, 2),
        "ctx_chars": len(example["context"]),
        "stats": stats,
    }
    record.update(example.get("scoring", {}))
    if not succeeded:
        error_type = getattr(result, "error_type", "RLMError")
        error = getattr(result, "error", "")
        record["error"] = f"{error_type}: {error}"
        record["pred"] = None
        record["predicted_answer"] = ""
        record["finished"] = False
    return record


def write_run_artifacts(checkpoint_path: Path, run_dir: Path, config: dict) -> dict:
    records = []
    for line in checkpoint_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    frame = pd.DataFrame(records)
    if "predicted_answer" not in frame.columns and "pred" in frame.columns:
        frame["predicted_answer"] = frame["pred"]
    if "id" in frame:
        frame = frame.drop_duplicates("id", keep="last").reset_index(drop=True)
    frame.to_csv(run_dir / "predictions.csv", index=False)

    errored = frame["error"].notna() if "error" in frame.columns else pd.Series(False, index=frame.index)
    scored_frame = frame[~errored]
    metrics: dict[str, Any]
    if len(scored_frame):
        metrics = score_prediction_frame("loft", scored_frame)
    else:
        metrics = {}

    count = len(frame)
    scored = len(scored_frame)
    unfinished = 0
    if scored and "finished" in scored_frame:
        unfinished = int((~scored_frame["finished"].astype(bool)).sum())
    metrics["runtime"] = {
        "examples": count,
        "scored": scored,
        "errors": int(errored.sum()),
        "average_tokens": float(scored_frame["tokens"].mean()) if scored and "tokens" in scored_frame else 0.0,
        "average_latency_s": float(scored_frame["latency_s"].mean()) if scored and "latency_s" in scored_frame else 0.0,
        "unfinished": unfinished,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return metrics


def build_rlm(args: argparse.Namespace) -> Any:
    ensure_recursive_llm_on_path()
    from rlm import RLM

    api_base = normalize_endpoint(args.endpoint) if args.endpoint else None
    api_key = args.api_key
    if api_key is None and args.endpoint:
        api_key = "EMPTY"

    llm_kwargs: dict[str, Any] = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if args.enable_thinking and args.endpoint:
        llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    if args.endpoint:
        install_vllm_reasoning_split()

    return RLM(
        model=litellm_model_id(args.model),
        recursive_model=litellm_model_id(args.model),
        api_base=api_base,
        api_key=api_key,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        repl_timeout=args.repl_timeout,
        max_output_chars=args.max_output_chars,
        max_elapsed_seconds=args.max_elapsed_seconds or None,
        max_retries=args.max_retries,
        capture_trajectory_content=bool(args.debug or args.save_trajectory),
        **llm_kwargs,
    )


def run_examples(
    rlm: Any,
    examples: Iterable[dict],
    *,
    include_answer_prefix: bool,
    checkpoint_path: Path,
    transcript_dir: Path,
    save_trajectory: bool,
    debug: bool,
    done: set[str],
) -> tuple[int, int]:
    n, finished = 0, 0
    with checkpoint_path.open("a") as fout:
        for example in examples:
            if example["id"] in done:
                continue
            query = build_query(example, include_answer_prefix)
            t0 = time.time()
            try:
                result = rlm.try_complete_result(query, example["context"])
                record = record_from_result(example, query, result, time.time() - t0)
            except Exception as exc:
                record = {
                    "id": example["id"],
                    "dataset": "loft",
                    "task": example.get("task") or "",
                    "mode": "recursive-llm",
                    "answers": example["answers"],
                    "answer": example.get("scoring", {}).get("answer", example["answers"]),
                    "answer_prefix": example.get("answer_prefix") or "",
                    "query": query,
                    "pred": None,
                    "predicted_answer": "",
                    "finished": False,
                    "end_reason": "exception",
                    "steps": 0,
                    "sub_calls": 0,
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": round(time.time() - t0, 2),
                    "ctx_chars": len(example["context"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                result = None
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            if save_trajectory and result is not None:
                (transcript_dir / f"{example['id'].replace(':', '_')}.json").write_text(
                    json.dumps(
                        {
                            "question": query,
                            "answers": example["answers"],
                            "pred": record.get("pred"),
                            "end_reason": record.get("end_reason"),
                            "stats": getattr(result, "stats", {}),
                            "trajectory": trajectory_as_dicts(result),
                        },
                        indent=2,
                    )
                )
            n += 1
            if record.get("finished"):
                finished += 1
            if debug:
                print(
                    f"  [{example['id']}] finished={record.get('finished')} "
                    f"end={record.get('end_reason')} pred={str(record.get('pred'))[:200]!r} "
                    f"t={record['latency_s']}s"
                )
            else:
                print(
                    f"  [{example['id']}] finished={record.get('finished')} "
                    f"end={record.get('end_reason')} tokens={record.get('tokens')} "
                    f"t={record['latency_s']}s"
                )
    return n, finished


def run_task(args: argparse.Namespace, task: str, rlm: Any) -> dict:
    components = [
        "loft",
        task,
        model_slug(args.model),
        "recursive-llm",
        f"split-{args.split}",
    ]
    if args.limit is not None:
        components.append(f"limit-{args.limit}")
    components.append("think" if args.enable_thinking else "nothink")
    run_dir = Path(args.out) / "__".join(components)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.jsonl"
    transcript_dir = run_dir / "transcripts"
    transcript_dir.mkdir(exist_ok=True)
    done = load_done(checkpoint_path)
    config = {
        "backend": "recursive-llm",
        "dataset": "loft",
        "data_dir": task,
        "model": args.model,
        "endpoint": normalize_endpoint(args.endpoint) if args.endpoint else None,
        "mode": "recursive-llm",
        "split": args.split,
        "limit": args.limit,
        "max_depth": args.max_depth,
        "max_iterations": args.max_iterations,
        "repl_timeout": args.repl_timeout,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "vllm_split_reasoning": bool(args.endpoint),
        "max_tokens": args.max_tokens,
        "recursive_llm": True,
        "rlm_query": True,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"== loft/{task} / recursive-llm -> {run_dir} ({len(done)} already done)")
    examples = load_examples(
        "loft",
        task,
        args.limit,
        args.split,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
    )
    n, finished = run_examples(
        rlm,
        examples,
        include_answer_prefix=not args.no_answer_prefix,
        checkpoint_path=checkpoint_path,
        transcript_dir=transcript_dir,
        save_trajectory=args.save_trajectory,
        debug=args.debug,
        done=done,
    )
    if n:
        print(f"== {task}: {finished}/{n} finished")
    if not checkpoint_path.exists() or not checkpoint_path.read_text().strip():
        print(f"== {task}: nothing to score")
        return {}
    metrics = write_run_artifacts(checkpoint_path, run_dir, config)
    print(f"== artifacts written to {run_dir}; metrics={metrics}")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    reexec_into_recursive_llm_env(argv)
    if not in_recursive_llm_env():
        raise SystemExit(f"refusing to run outside {recursive_llm_python()} (current interpreter is {sys.executable})")
    args = parse_args(argv)
    print(f"using recursive-llm env {sys.executable}")
    for env_file in load_env_files():
        print(f"loaded provider keys from {env_file}")
    if "loft" not in SCORER_REGISTRY:
        raise SystemExit("LOFT scorer is missing from the evaluation registry")
    try:
        tasks = loft_128k_tasks(args.task)
    except ValueError as exc:
        raise SystemExit(str(exc))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rlm = build_rlm(args)
    target_endpoint = normalize_endpoint(args.endpoint) if args.endpoint else "direct API"
    print(
        f"recursive-llm -> {target_endpoint} "
        f"model={litellm_model_id(args.model)} max_depth={args.max_depth} "
        f"thinking={'on' if args.enable_thinking else 'off'} "
        f"(REPL exposes rlm_query / recursive_llm)"
    )
    for task in tasks:
        run_task(args, task, rlm)


if __name__ == "__main__":
    main()
