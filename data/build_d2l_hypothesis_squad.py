import argparse
import json
import os
import urllib.request
from pathlib import Path

from datasets import load_dataset


ANSWER_ONLY_INSTRUCTION = (
    "Output only the answer and do not output any other words."
)
FULL_SENTENCE_INSTRUCTION = (
    'Answer with a complete sentence in exactly this form: "The answer is <answer>."'
)

INSTRUCTION_TYPES = ("answer_only", "full_sentence")
INSTRUCTIONS = {
    "answer_only": ANSWER_ONLY_INSTRUCTION,
    "full_sentence": FULL_SENTENCE_INSTRUCTION,
}
CONDITIONS = (
    "content_adapter",
    "content_wrong",
    "instruction_adapter",
    "instruction_wrong",
    "prompt_only",
)
SQUAD_V1_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def format_response(answer: str, instruction_type: str) -> str:
    if instruction_type == "answer_only":
        return answer
    if instruction_type == "full_sentence":
        return f"The answer is {answer}."
    raise ValueError(f"Unsupported instruction type: {instruction_type}")


def build_prompt(
    passage: str,
    question: str,
    instruction: str,
    condition: str,
) -> str:
    if condition.startswith("content_"):
        return f"Passage:\n{passage}\n\n{instruction}"
    if condition.startswith("instruction_"):
        return f"Passage:\n{passage}\n\nQuestion: {question}"
    if condition == "prompt_only":
        return f"Passage:\n{passage}\n\nQuestion: {question}\n\n{instruction}"
    raise ValueError(f"Unsupported condition: {condition}")


def get_context(
    examples: list[dict],
    idx: int,
    condition: str,
    instruction_type: str,
) -> str:
    wrong_idx = (idx + 1) % len(examples)
    if condition == "content_adapter":
        return examples[idx]["question"]
    if condition == "content_wrong":
        return examples[wrong_idx]["question"]
    if condition == "instruction_adapter":
        return INSTRUCTIONS[instruction_type]
    if condition == "instruction_wrong":
        wrong_type = (
            "full_sentence"
            if instruction_type == "answer_only"
            else "answer_only"
        )
        return INSTRUCTIONS[wrong_type]
    if condition == "prompt_only":
        return ""
    raise ValueError(f"Unsupported condition: {condition}")


def flatten_squad_v1(data: dict) -> list[dict]:
    samples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa["answers"]
                samples.append(
                    {
                        "context": context,
                        "question": qa["question"],
                        "answers": {"text": [answer["text"] for answer in answers]},
                    }
                )
    return samples


def load_squad_samples(source: str, split: str):
    try:
        return list(load_dataset(source, split=split))
    except Exception as e:
        if source not in {"rajpurkar/squad", "squad"} or split not in SQUAD_V1_URLS:
            raise
        print(
            "Falling back to official SQuAD v1.1 JSON because "
            f"datasets.load_dataset failed: {type(e).__name__}: {e}"
        )
        with urllib.request.urlopen(SQUAD_V1_URLS[split]) as response:
            data = json.loads(response.read().decode("utf-8"))
        return flatten_squad_v1(data)


def load_squad(source: str, split: str, n_samples: int, seed: int) -> list[dict]:
    ds = load_squad_samples(source, split)
    if n_samples > len(ds):
        raise ValueError(f"Requested {n_samples} samples, but split has {len(ds)}")

    # Match datasets.Dataset.shuffle(seed).select(range(n)) without requiring the
    # source to be a Dataset object after the JSON fallback.
    import random

    rng = random.Random(seed)
    ds = list(ds)
    rng.shuffle(ds)
    ds = ds[:n_samples]

    examples = []
    for idx, sample in enumerate(ds):
        answers = sample["answers"]["text"]
        if not answers:
            continue
        instruction_type = INSTRUCTION_TYPES[idx % len(INSTRUCTION_TYPES)]
        examples.append(
            {
                "source_index": idx,
                "passage": sample["context"],
                "question": sample["question"],
                "gold_answer": answers[0],
                "all_answers": answers,
                "instruction_type": instruction_type,
                "instruction": INSTRUCTIONS[instruction_type],
            }
        )
    return examples


def build_rows(examples: list[dict], condition: str) -> list[dict]:
    rows = []
    for idx, example in enumerate(examples):
        instruction_type = example["instruction_type"]
        prompt = build_prompt(
            passage=example["passage"],
            question=example["question"],
            instruction=example["instruction"],
            condition=condition,
        )
        context = get_context(
            examples=examples,
            idx=idx,
            condition=condition,
            instruction_type=instruction_type,
        )
        wrong_context_source_index = -1
        if condition in {"content_wrong", "instruction_wrong"}:
            wrong_context_source_index = (idx + 1) % len(examples)

        rows.append(
            {
                "context": context,
                "prompts": [prompt],
                "responses": [
                    format_response(example["gold_answer"], instruction_type)
                ],
                "condition": condition,
                "instruction_type": instruction_type,
                "instruction": example["instruction"],
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "all_answers": example["all_answers"],
                "source_index": example["source_index"],
                "wrong_context_source_index": wrong_context_source_index,
            }
        )
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SQuAD pilot datasets for D2L content/instruction transfer."
    )
    parser.add_argument(
        "--source",
        default="data/raw_datasets/squad",
        help="SQuAD dataset path or Hugging Face dataset name.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="data/raw_datasets/d2l_hypothesis/squad",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    examples = load_squad(
        source=args.source,
        split=args.split,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    if len(examples) != args.n_samples:
        raise ValueError(
            f"Expected {args.n_samples} usable examples, found {len(examples)}"
        )

    output_dir = Path(args.output_dir)
    for condition in CONDITIONS:
        rows = build_rows(examples, condition)
        out_path = output_dir / condition / f"{args.split}.jsonl"
        write_jsonl(rows, out_path)
        print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
