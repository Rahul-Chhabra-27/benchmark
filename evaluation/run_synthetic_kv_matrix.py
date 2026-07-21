"""Run the synthetic-KV 64K budget matrix with one Qwen model load."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


LOFT_MEMORY_BUDGETS = [
    (256.0, "MB"),
    (512.0, "MB"),
    (1.0, "GB"),
    (2.0, "GB"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="evaluate_synthetic_kv_config.yaml")
    parser.add_argument(
        "--max-memory-budgets",
        type=int,
        default=None,
        help="Run only the first N explicit memory budgets; the baseline is still included.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_values = asdict(EvaluationConfig())
    config_values.update(_load_yaml_config(args.config_file))
    config = EvaluationConfig(**config_values)

    if config.dataset != "synthetic_kv":
        raise ValueError(f"Expected dataset='synthetic_kv', got {config.dataset!r}")
    if config.data_dir != ["64k"]:
        raise ValueError("Synthetic-KV matrix config requires data_dir: ['64k']")

    memory_budgets = LOFT_MEMORY_BUDGETS
    if args.max_memory_budgets is not None:
        if args.max_memory_budgets < 1:
            raise ValueError("--max-memory-budgets must be positive")
        memory_budgets = memory_budgets[: args.max_memory_budgets]

    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=["64k"],
        memory_budgets=memory_budgets,
        baseline_compression_ratio=0.01,
        include_baseline=True,
    )


if __name__ == "__main__":
    main()
