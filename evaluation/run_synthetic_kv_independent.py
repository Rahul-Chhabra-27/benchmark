"""Run one isolated Synthetic-KV memory-budget configuration."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


CONFIGURATIONS = [
    (None, "MB", "baseline"),
    (512.0, "MB", "512MB"),
    (1.0, "GB", "1GB"),
    (2.0, "GB", "2GB"),
    (4.0, "GB", "4GB"),
    (256.0, "MB", "256MB"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="evaluate_synthetic_kv_independent_config.yaml",
    )
    parser.add_argument("--configuration-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.configuration_id < len(CONFIGURATIONS):
        raise ValueError(
            f"configuration-id must be between 0 and {len(CONFIGURATIONS) - 1}"
        )

    config_values = asdict(EvaluationConfig())
    config_values.update(_load_yaml_config(args.config_file))
    config = EvaluationConfig(**config_values)

    if config.dataset != "synthetic_kv":
        raise ValueError(f"Expected dataset='synthetic_kv', got {config.dataset!r}")
    if config.data_dir != ["64k"]:
        raise ValueError("Synthetic-KV independent config requires data_dir: ['64k']")
    if config.fraction != 1.0:
        raise ValueError("Synthetic-KV independent runs require fraction: 1.0")

    memory_budget, memory_budget_unit, label = CONFIGURATIONS[args.configuration_id]
    print(
        f"Independent configuration {args.configuration_id}: {label}; "
        f"fraction={config.fraction}",
        flush=True,
    )

    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=["64k"],
        memory_budgets=(
            [] if memory_budget is None else [(memory_budget, memory_budget_unit)]
        ),
        baseline_compression_ratio=0.01,
        include_baseline=memory_budget is None,
    )


if __name__ == "__main__":
    main()
