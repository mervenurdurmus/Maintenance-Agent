import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

DATASET_PATH = Path("evaluation/golden/golden_dataset.jsonl")
REPORT_PATH = Path("evaluation/reports/ragas_summary.json")


def load_dataset() -> Dataset:
    rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )


def main() -> None:
    dataset = load_dataset()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
