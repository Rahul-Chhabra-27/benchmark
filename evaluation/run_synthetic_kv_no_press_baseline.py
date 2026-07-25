"""Run the full non-quantized Synthetic-KV 64K true no-press baseline."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="evaluate_synthetic_kv_independent_config.yaml",
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
        raise ValueError("Synthetic-KV no-press baseline requires data_dir: ['64k']")
    if config.fraction != 1.0:
        raise ValueError("Synthetic-KV no-press baseline requires fraction: 1.0")
    if config.int8 or config.int4 or config.fp8:
        raise ValueError("Synthetic-KV no-press baseline must be non-quantized")

    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=["64k"],
        memory_budgets=[],
        baseline_compression_ratio=0.0,
        include_baseline=True,
        baseline_press_name="no_press",
    )


if __name__ == "__main__":
    main()
