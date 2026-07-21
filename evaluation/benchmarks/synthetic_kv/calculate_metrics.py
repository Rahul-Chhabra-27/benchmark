"""Metrics for exact synthetic key/value retrieval."""

import re
from collections.abc import Iterable

import pandas as pd


CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f]")


def _normalize(value: object) -> str:
    return CONTROL_CHARACTERS.sub("", str(value)).strip().casefold()


def _references(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [_normalize(value)]
    return [_normalize(reference) for reference in value]


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Report strict exact match and answer containment percentages."""
    if len(df) == 0:
        raise ValueError("Cannot score an empty synthetic-KV dataframe")

    exact_matches = 0
    string_matches = 0
    for prediction, answer in zip(df["predicted_answer"], df["answer"]):
        normalized_prediction = _normalize(prediction)
        references = _references(answer)
        exact_matches += int(normalized_prediction in references)
        string_matches += int(any(reference in normalized_prediction for reference in references))

    sample_count = len(df)
    return {
        "synthetic_kv_64k": {
            "exact_match": round(100.0 * exact_matches / sample_count, 2),
            "string_match": round(100.0 * string_matches / sample_count, 2),
            "num_samples": sample_count,
        }
    }
