"""Run a worker's RULER 64K task/budget matrix with one model load."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


RULER_64K_BUDGETS = [
    (512.0, "MB"),
    (1.0, "GB"),
    (2.0, "GB"),
    (3.0, "GB"),
    (4.0, "GB"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="evaluate_ruler64k_config.yaml")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_count < 1:
        raise ValueError("worker-count must be positive")
    if not 0 <= args.worker_id < args.worker_count:
        raise ValueError("worker-id must be between 0 and worker-count - 1")

    config_values = asdict(EvaluationConfig())
    config_values.update(_load_yaml_config(args.config_file))
    if args.fraction is not None:
        config_values["fraction"] = args.fraction
    config = EvaluationConfig(**config_values)

    if config.dataset != "ruler64k":
        raise ValueError(f"Expected dataset='ruler64k', got {config.dataset!r}")
    if not isinstance(config.data_dir, list):
        raise ValueError("RULER 64K matrix config requires data_dir to be a task list")

    assigned_tasks = config.data_dir[args.worker_id :: args.worker_count]
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise ValueError("max-tasks must be positive")
        assigned_tasks = assigned_tasks[: args.max_tasks]
    print(
        f"Worker {args.worker_id}/{args.worker_count} assigned tasks: "
        f"{', '.join(assigned_tasks)}"
    )

    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=assigned_tasks,
        memory_budgets=RULER_64K_BUDGETS,
        baseline_compression_ratio=0.01,
    )


if __name__ == "__main__":
    main()
