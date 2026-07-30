"""Run one full Synthetic-KV no-prefix KVzip memory budget."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


MEMORY_BUDGETS_MB = {512, 1024, 2048, 4096, 8192}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="evaluate_synthetic_kv_no_prefix_budgets.yaml",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=int,
        choices=sorted(MEMORY_BUDGETS_MB),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_values = asdict(EvaluationConfig())
    config_values.update(_load_yaml_config(args.config_file))
    config = EvaluationConfig(**config_values)

    if config.dataset != "synthetic_kv" or config.data_dir != ["64k"]:
        raise ValueError("Expected the full Synthetic-KV 64K dataset")
    if config.fraction != 1.0:
        raise ValueError("No-prefix budget runs require fraction=1.0")
    if config.synthetic_kv_prefix_mode != "strip":
        raise ValueError("No-prefix budget runs require synthetic_kv_prefix_mode='strip'")
    if config.press_name != "kvzip":
        raise ValueError("Memory-budget runs require press_name='kvzip'")
    if config.int8 or config.int4 or config.fp8:
        raise ValueError("These runs must use the non-quantized model")

    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=["64k"],
        memory_budgets=[(float(args.memory_budget_mb), "MB")],
        baseline_compression_ratio=0.01,
        include_baseline=False,
    )


if __name__ == "__main__":
    main()
