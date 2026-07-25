"""Run a worker's true no-press LOFT baseline and no budgeted configurations."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_count < 1:
        raise ValueError("worker-count must be positive")
    if not 0 <= args.worker_id < args.worker_count:
        raise ValueError("worker-id must be between 0 and worker-count - 1")

    config_values = asdict(EvaluationConfig())
    config_values.update(_load_yaml_config(args.config_file))
    config = EvaluationConfig(**config_values)
    if config.dataset != "loft":
        raise ValueError(f"Expected dataset='loft', got {config.dataset!r}")
    if not isinstance(config.data_dir, list):
        raise ValueError("LOFT baseline config requires data_dir to be a task list")

    assigned_tasks = config.data_dir[args.worker_id :: args.worker_count]
    print(
        f"No-press worker {args.worker_id}/{args.worker_count} assigned tasks: "
        f"{', '.join(assigned_tasks)}"
    )
    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=assigned_tasks,
        memory_budgets=[],
        baseline_compression_ratio=0.0,
        include_baseline=True,
        baseline_press_name="no_press",
    )


if __name__ == "__main__":
    main()
