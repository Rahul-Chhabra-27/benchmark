"""Run a five-question non-quantized Synthetic-KV no-press smoke test."""

import argparse
from dataclasses import asdict

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="evaluate_synthetic_kv_no_prefix_baseline_smoke5.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_values = asdict(EvaluationConfig())
    config_values.update(_load_yaml_config(args.config_file))
    config = EvaluationConfig(**config_values)

    if config.dataset != "synthetic_kv" or config.data_dir != ["64k"]:
        raise ValueError("Expected the Synthetic-KV 64K dataset")
    if config.press_name != "no_press":
        raise ValueError("Smoke test requires press_name='no_press'")
    if config.int8 or config.int4 or config.fp8:
        raise ValueError("Smoke test must use the non-quantized model")

    runner = EvaluationRunner(config)
    runner.run_evaluation()


if __name__ == "__main__":
    main()
