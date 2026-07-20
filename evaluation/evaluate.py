# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import BitsAndBytesConfig, FineGrainedFP8Config, Pipeline, pipeline
from verify_int8_model import verify_int8_model

from kvpress import (
    ComposedPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    ThinKPress,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    # data_dir: Optional[str] = None
    data_dir: Optional[Union[str, List[str]]] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None
    head_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None
    memory_budget: Optional[float] = None
    memory_budget_unit: str = "GB"
    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    query_aware: bool = False
    needle_depth: Optional[int] = None

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    # Quantization
    fp8: bool = False
    int8: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        if self.memory_budget is not None:
            assert self.memory_budget > 0, f"memory_budget must be positive, got {self.memory_budget}"
            self.memory_budget_unit = self.memory_budget_unit.upper()
            assert self.memory_budget_unit in {"MB", "GB"}, (
                f"memory_budget_unit must be MB or GB, got {self.memory_budget_unit}"
            )

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        assert not (self.fp8 and self.int8), "fp8 and int8 quantization cannot both be enabled"

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self, output_dir: Path,data_dir: Optional[str] = None) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        # Convert list to string for directory name
        if isinstance(data_dir, list):
            data_dir_str = "__".join(data_dir)
        else:
            data_dir_str = str(data_dir) if data_dir else ""
        # Build directory name components
        components = [
            self.dataset,
            data_dir_str,
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        if self.threshold is not None:
            components[-1] = f"{self.threshold:.2f}"
        elif self.head_compression_ratio is not None:
            components[-1] = f"{self.head_compression_ratio:.2f}"
        if self.memory_budget is not None:
            components.append(f"memory_budget{self.memory_budget:g}{self.memory_budget_unit}")
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.int8:
            components.append("int8")
        if self.query_aware:
            components.append("query_aware")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        dir_name = f"new_{dir_name}" 
        config_dir = output_dir / dir_name

        # Use a deterministic directory so interrupted matrix runs can resume.
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        config_dict = asdict(self)
        if self.threshold is not None or self.head_compression_ratio is not None:
            config_dict.pop("compression_ratio", None)
        if self.threshold is None:
            config_dict.pop("threshold", None)
        if self.head_compression_ratio is None:
            config_dict.pop("head_compression_ratio", None)
        with open(str(config_filename), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            assert (
                self.config.head_compression_ratio is not None
            ), "head_compression_ratio must be set for DuoAttentionPress"
            press.head_compression_ratio = self.config.head_compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {press.head_compression_ratio}")
        elif isinstance(press, DMSPress):
            assert self.config.threshold is not None, "threshold must be set for DMSPress"
            press.threshold = self.config.threshold
            logger.info(f"Set DMSPress threshold to {press.threshold}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, DecodingPress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")
    def _load_datasets_infinite_bench(self,task_data_dir: List[str]) -> pd.DataFrame:
        """Load InfiniteBench datasets by individual configs.
        
        InfiniteBench requires loading each subset as a separate config.
        
        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        benchmark_name: str = "infinite_bench"
        huggingface_dataset_id: str = "MaxJeblick/InfiniteBench"
        print(f"Loading InfiniteBench datasets: {task_data_dir}")
        dfs = []
        
        for subset in task_data_dir:
            try:
                from datasets import load_dataset
                subset_dataset = load_dataset(huggingface_dataset_id, subset, split="test")
                subset_df = subset_dataset.to_pandas()
                subset_df["task"] = subset  # Ensure task column exists
                dfs.append(subset_df)
                print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue
        
        if not dfs:
            raise Exception("No InfiniteBench subsets could be loaded successfully")
        
        # Combine all subset DataFrames
        import pandas as pd
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        return combined_df
    def _load_datasets_ruler32k(self, task_data_dir: List[str]) -> pd.DataFrame:
        """Load Ruler datasets by individual configs.

        Ruler requires loading each context length as a separate config.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading Ruler datasets: {task_data_dir}")
        dfs = []
        benchmark_name: str = "ruler32k"
        huggingface_dataset_id: str = "xAlg-AI/att-hub-ruler-32k"
        for subset in task_data_dir:
            try:
                from datasets import load_dataset

                subset_dataset = load_dataset(
                    huggingface_dataset_id, subset, split=subset
                )
                subset_df = subset_dataset.to_pandas()
                # Add context length as a column for analysis
                subset_df["context_length"] = 32768
                dfs.append(subset_df)
                print(
                    f"  ✓ Loaded {len(subset_df)} samples from {subset} || context_length = 32768"
                )
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue

        if not dfs:
            raise Exception("No Ruler subsets could be loaded successfully")

        # Combine all subset DataFrames
        import pandas as pd

        combined_df = pd.concat(dfs, ignore_index=True)
        print(
            f"Combined {len(combined_df)} total samples from {len(dfs)} subsets || context_length = 32768"
        )
        return combined_df

    def _load_dataset_ruler64k(self, task: str) -> pd.DataFrame:
        """Load one task from the cached Qwen-tokenized RULER 64K dataset."""
        huggingface_dataset_id = DATASET_REGISTRY["ruler64k"]
        dataset = load_dataset(
            huggingface_dataset_id,
            "65536",
            split="test",
        )

        available_tasks = set(dataset.unique("task"))
        if task not in available_tasks:
            available = ", ".join(sorted(available_tasks))
            raise ValueError(f"Unknown RULER 64K task {task!r}. Available tasks: {available}")

        task_dataset = dataset.filter(lambda example: example["task"] == task)
        task_df = task_dataset.to_pandas()
        task_df["context_length"] = 65536
        print(f"  ✓ Loaded {len(task_df)} samples from {task} || target context length = 65536")
        return task_df
     ### TODO : specially used for loft rag dataset
    def _load_datasets(self, task_data_dir: List[str]) -> pd.DataFrame:
        """Load LOFT RAG datasets from HuggingFace Hub.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        
        print(f"Loading LOFT RAG datasets: {task_data_dir}")
        dfs: List[pd.DataFrame] = []

        for subset in task_data_dir:
            parts: List[str] = subset.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid subset format: {subset} (expected: dataset_length)"
                )

            length: str = parts[-1]
            dataset: str = "_".join(parts[:-1])
            hf_dataset_id: str = f"f20180301/loft-rag-{dataset}-{length}"

            dataset_dict = load_dataset(hf_dataset_id)

            subset_dfs: List[pd.DataFrame] = []
            for split_name in ["dev", "test"]:
                if split_name in dataset_dict:
                    split_df: pd.DataFrame = dataset_dict[split_name].to_pandas()
                    split_df["split"] = split_name
                    subset_dfs.append(split_df)

            if not subset_dfs:
                raise ValueError(f"No splits found for {subset} ({hf_dataset_id})")

            subset_df: pd.DataFrame = pd.concat(subset_dfs, ignore_index=True)
            subset_df["task"] = subset
            dfs.append(subset_df)
            print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")

        if not dfs:
            raise ValueError("No LOFT RAG subsets could be loaded")

        combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")

        required_columns: List[str] = [
            "context",
            "question",
            "answers",
            "task",
            "answer_prefix",
            "max_new_tokens",
        ]
        missing_columns: List[str] = [
            col for col in required_columns if col not in combined_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return combined_df
    def _load_and_prepare_dataset(self,task_data_dir: Optional[str] = None):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        if dataset_name == "infinitebench":
            try:
                task_data_dir = [task_data_dir]
                df = self._load_datasets_infinite_bench(task_data_dir)
            except Exception as e:
                logger.error(f"Failed to load InfiniteBench dataset: {e}")
                raise
        elif dataset_name == "loft":
            try:
                task_data_dir = [task_data_dir]
                df = self._load_datasets(task_data_dir)
            except Exception as e:
                logger.error(f"Failed to load LOFT RAG dataset: {e}")
                raise
        elif dataset_name == "ruler32k":
            try:
                task_data_dir = [task_data_dir]
                df = self._load_datasets_ruler32k(task_data_dir)
            except Exception as e:
                logger.error(f"Failed to load Ruler32k dataset: {e}")
                raise
        elif dataset_name == "ruler64k":
            if task_data_dir is None:
                raise ValueError("RULER 64K requires a task name in data_dir")
            try:
                df = self._load_dataset_ruler64k(task_data_dir)
            except Exception as e:
                logger.error(f"Failed to load RULER 64K task {task_data_dir!r}: {e}")
                raise
        else:
        # data_dir = str(self.config.data_dir) if self.config.data_dir else None
            data_dir = task_data_dir if task_data_dir is not None else (
                str(self.config.data_dir) if self.config.data_dir else None
            )
            logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
            df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()
        fraction = self.config.fraction
        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        if isinstance(self.press, FinchPress):
            if not self.config.query_aware:
                logger.error("FinchPress requires 'query_aware' to be set to True.")
                raise ValueError("FinchPress requires query_aware to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.query_aware:
            logger.info("Query-aware compression: including question in context for compression.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}

        if self.config.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
            logger.info("FP8 quantization enabled.")

        if self.config.int8:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            logger.info("INT8 bitsandbytes quantization enabled.")

        if isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)

        if self.config.int8:
            int8_verification = verify_int8_model(self.pipeline.model)
            logger.info("INT8 model verification passed: %s", int8_verification)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]

        if isinstance(self.press, DecodingPress):
            logger.info("DecodingPress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                questions = df_group["question"].to_list()
                # Use max_new_tokens from config, or fallback to dataset's default for the task
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]

                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                    memory_budget=self.config.memory_budget,
                    memory_budget_unit=self.config.memory_budget_unit,
                )
                self.df.loc[df_group.index, "predicted_answer"] = output["answers"]  # type: ignore[union-attr]
                budget_stats = getattr(self.pipeline, "last_memory_budget_stats", None)
                if budget_stats is not None:
                    for column, value in budget_stats.items():
                        self.df.loc[df_group.index, column] = value
                else:
                    self.df.loc[df_group.index, "compression_ratio"] = (
                        self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                    )  # type: ignore[union-attr, attr-defined]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        score = scorer(self.df)  # type: ignore[call-arg]
        metrics = score if isinstance(score, dict) else {"score": score}

        if "compression_ratio" in self.df.columns:
            # A shared context may have multiple questions. Count that context once in the summary.
            context_stats = self.df.drop_duplicates(subset=["context"])
            metrics["average_compression_ratio"] = float(context_stats["compression_ratio"].mean())
            metrics["average_original_context_tokens"] = float(context_stats["context_tokens"].mean())
            metrics["average_retained_context_tokens"] = float(
                context_stats["retained_context_tokens"].mean()
            )
            metrics["kv_memory_per_token_kb"] = float(context_stats["kv_memory_per_token_kb"].iloc[0])
            metrics["average_retained_kv_memory_mb"] = float(
                context_stats["retained_kv_memory_mb"].mean()
            )
            metrics["average_retained_kv_memory_gb"] = float(
                context_stats["retained_kv_memory_gb"].mean()
            )
            metrics["average_uncompressed_kv_memory_mb"] = float(
                context_stats["uncompressed_kv_memory_mb"].mean()
            )
            metrics["average_uncompressed_kv_memory_gb"] = float(
                context_stats["uncompressed_kv_memory_gb"].mean()
            )

            if self.config.memory_budget is not None:
                metrics["memory_budget"] = self.config.memory_budget
                metrics["memory_budget_unit"] = self.config.memory_budget_unit
                metrics["token_budget"] = int(context_stats["token_budget"].iloc[0])

            logger.info(
                "Average retained context KV memory: "
                f"{metrics['average_retained_kv_memory_mb']:.2f} MB "
                f"({metrics['average_retained_kv_memory_gb']:.4f} GB); "
                f"average compression ratio: {metrics['average_compression_ratio']:.6f}"
            )

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")
        return metrics

    def _save_results_readme(self, readme_filename: Path, task: str, metrics: Dict[str, Any]):
        """Write a self-contained summary as soon as one task finishes."""
        if self.config.memory_budget is None:
            configuration = f"KVzip baseline (compression ratio {self.config.compression_ratio:.4f})"
        else:
            configuration = f"KVzip memory budget: {self.config.memory_budget:g} {self.config.memory_budget_unit}"

        metric_rows = "\n".join(
            f"| `{name}` | {value:.6f} |" if isinstance(value, float) else f"| `{name}` | {value} |"
            for name, value in metrics.items()
        )
        contents = f"""# {self.config.dataset.upper()} Benchmark Result

- Model: `{self.config.model}`
- Task: `{task}`
- Configuration: {configuration}
- Press: `{self.config.press_name}`
- Dataset fraction: `{self.config.fraction}`

## Metrics and KV-cache statistics

| Field | Value |
|---|---:|
{metric_rows}

Files in this directory:

- `predictions.csv`: per-sample predictions and KV-cache statistics
- `metrics.json`: machine-readable metrics and averages
- `config.yaml`: complete evaluation configuration
"""
        readme_filename.write_text(contents)
        logger.info(f"Result summary saved to {readme_filename}")

    def _reset_reused_model_state(self) -> None:
        """Clear state that must not leak between matrix configurations."""
        if self.pipeline is None:
            return

        self.pipeline.last_memory_budget_stats = None  # type: ignore[attr-defined]
        language_model = (
            self.pipeline.model.model.language_model
            if hasattr(self.pipeline.model.model, "language_model")
            else self.pipeline.model.model
        )
        for layer in language_model.layers:
            layer.self_attn.masked_key_indices = None

        if self.press is not None and hasattr(self.press, "_reset_internal_parameters"):
            self.press._reset_internal_parameters()  # type: ignore[attr-defined]

        self._setup_deterministic_seeds()
        torch.cuda.empty_cache()

    def run_memory_budget_matrix(
        self,
        tasks: list[str],
        memory_budgets: list[tuple[float, str]],
        baseline_compression_ratio: float = 0.01,
        include_baseline: bool = True,
    ) -> None:
        """Run multiple tasks and KV budgets while loading the model only once."""
        if not tasks:
            raise ValueError("At least one task is required for a matrix evaluation")

        output_dir = self._setup_directories()
        self.config.compression_ratio = baseline_compression_ratio
        self.config.memory_budget = None
        self._setup_press()
        self._setup_model_pipeline()

        configurations: list[tuple[Optional[float], str]] = list(memory_budgets)
        if include_baseline:
            configurations.insert(0, (None, "MB"))

        for task in tasks:
            logger.info(f"=== Starting matrix task: '{task}' ===")
            pending_configurations: list[tuple[Optional[float], str, Path]] = []

            for memory_budget, memory_budget_unit in configurations:
                self.config.data_dir = task
                self.config.compression_ratio = baseline_compression_ratio
                self.config.memory_budget = memory_budget
                self.config.memory_budget_unit = memory_budget_unit.upper()
                results_dir = self.config.get_results_dir(output_dir, data_dir=task)
                predictions_filename = results_dir / "predictions.csv"
                metrics_filename = results_dir / "metrics.json"

                if predictions_filename.exists() and metrics_filename.exists():
                    logger.info(
                        f"Completed results already exist for task={task}, "
                        f"memory_budget={memory_budget}{memory_budget_unit}; skipping."
                    )
                    continue
                pending_configurations.append((memory_budget, memory_budget_unit, results_dir))

            if not pending_configurations:
                logger.info(f"All matrix configurations already exist for task '{task}'; skipping dataset load.")
                continue

            # Dataset text is loaded and prepared once, then copied before every
            # inference configuration because scoring mutates predicted_answer.
            self.config.memory_budget = None
            self.config.memory_budget_unit = "MB"
            self.config.compression_ratio = baseline_compression_ratio
            self._setup_press()
            self._load_and_prepare_dataset(task_data_dir=task)
            source_df = self.df.copy(deep=True)  # type: ignore[union-attr]

            for memory_budget, memory_budget_unit, results_dir in pending_configurations:
                self.config.data_dir = task
                self.config.compression_ratio = baseline_compression_ratio
                self.config.memory_budget = memory_budget
                self.config.memory_budget_unit = memory_budget_unit.upper()
                self._setup_press()
                self._reset_reused_model_state()
                self.df = source_df.copy(deep=True)

                if memory_budget is None:
                    logger.info(
                        f"Running task={task}, KVzip reference "
                        f"compression_ratio={baseline_compression_ratio:.4f}"
                    )
                else:
                    logger.info(
                        f"Running task={task}, logical KVzip budget="
                        f"{memory_budget:g}{self.config.memory_budget_unit}"
                    )

                predictions_filename = results_dir / "predictions.csv"
                metrics_filename = results_dir / "metrics.json"
                config_filename = results_dir / "config.yaml"
                readme_filename = results_dir / "README.md"

                self._run_inference()
                self._save_results(predictions_filename)
                metrics = self._calculate_and_save_metrics(metrics_filename)
                self.config.save_config(config_filename)
                self._save_results_readme(readme_filename, task, metrics)
                logger.info(
                    f"Completed task={task}, memory_budget="
                    f"{memory_budget if memory_budget is not None else 'reference'}"
                    f"{self.config.memory_budget_unit if memory_budget is not None else ''}"
                )

            self.df = None
            del source_df
            torch.cuda.empty_cache()
            logger.info(f"=== Completed matrix task: '{task}' ===")

        logger.info("Memory-budget matrix evaluation completed successfully with one model load.")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()
        # Define all LongBench tasks
        longbench_tasks = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
            "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
            "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count",
            "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
        ]
        # Determine which tasks to run
        if self.config.data_dir is None or (isinstance(self.config.data_dir, list) and len(self.config.data_dir) == 0):
                # Run all LongBench tasks
                tasks_to_run = longbench_tasks
                logger.info(f"No specific tasks provided. Running all {len(tasks_to_run)} LongBench tasks.")
        else:
            # Run specific tasks
            if isinstance(self.config.data_dir, str):
                tasks_to_run = [self.config.data_dir]
            else:
                tasks_to_run = self.config.data_dir
            logger.info(f"Running specific tasks: {tasks_to_run}")
        self._setup_press()
        self._setup_model_pipeline()

        for task in tasks_to_run:
            logger.info(f"Starting evaluation for task: {task}")    
            results_dir = self.config.get_results_dir(output_dir, data_dir=task)
            predictions_filename = results_dir / "predictions.csv"
            metrics_filename = results_dir / "metrics.json"
            config_filename = results_dir / "config.yaml"
            readme_filename = results_dir / "README.md"

            if predictions_filename.exists() and metrics_filename.exists():
                logger.info(
                    f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
                )
                continue

            
            self._load_and_prepare_dataset(task_data_dir=task)

            self._run_inference()
            self._save_results(predictions_filename)
            metrics = self._calculate_and_save_metrics(metrics_filename)
            self.config.save_config(config_filename)
            self._save_results_readme(readme_filename, task, metrics)
            logger.info(f"=== Completed task: '{task}' ===")
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
