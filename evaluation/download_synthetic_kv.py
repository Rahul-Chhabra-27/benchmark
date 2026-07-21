"""Cache the public synthetic-KV dataset on an internet-enabled login node."""

from datasets import load_dataset


DATASET_ID = "ollamaweights/synthetic-kv-qwen3-8b"


def main() -> None:
    dataset = load_dataset(DATASET_ID, split="test")
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one compact context, found {len(dataset)}")

    row = dataset[0]
    if len(row["questions"]) != len(row["answers"]):
        raise RuntimeError("Dataset questions and answers have different lengths")

    print(
        f"Cached {DATASET_ID}/64k: contexts={len(dataset)}, "
        f"questions={len(row['questions'])}, context_tokens={row['context_tokens']}"
    )


if __name__ == "__main__":
    main()
