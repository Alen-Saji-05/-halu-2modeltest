"""Preprocess HaluEval dataset into LettuceDetect's HallucinationData format.

HaluEval provides example-level labels (right_answer vs hallucinated_answer).
We convert this to LettuceDetect's span-level format by:
- Right answers: labels = [] (no hallucination spans)
- Hallucinated answers: labels = [{start: 0, end: len(answer)}] (entire answer is hallucinated)
"""

import argparse
import json
import random
from pathlib import Path

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample


def process_qa_data(data: list[dict], split: str) -> list[HallucinationSample]:
    """Process HaluEval QA data into HallucinationSamples."""
    samples = []
    for item in data:
        knowledge = item.get("knowledge", "")
        question = item.get("question", "")
        prompt = f"{knowledge}\n\nQuestion: {question}"

        # Right answer → supported (no hallucination labels)
        right_answer = item.get("right_answer", "")
        if right_answer:
            samples.append(HallucinationSample(
                prompt=prompt,
                answer=right_answer,
                labels=[],
                split=split,
                task_type="qa",
                dataset="halueval",
                language="en",
            ))

        # Hallucinated answer → entire answer is hallucinated
        hallucinated_answer = item.get("hallucinated_answer", "")
        if hallucinated_answer:
            samples.append(HallucinationSample(
                prompt=prompt,
                answer=hallucinated_answer,
                labels=[{"start": 0, "end": len(hallucinated_answer), "label": "hallucination"}],
                split=split,
                task_type="qa",
                dataset="halueval",
                language="en",
            ))

    return samples


def process_dialogue_data(data: list[dict], split: str) -> list[HallucinationSample]:
    """Process HaluEval dialogue data into HallucinationSamples."""
    samples = []
    for item in data:
        knowledge = item.get("knowledge", "")
        dialogue_history = item.get("dialogue_history", "")
        prompt = f"{knowledge}\n\nDialogue History: {dialogue_history}"

        right_response = item.get("right_response", "")
        if right_response:
            samples.append(HallucinationSample(
                prompt=prompt,
                answer=right_response,
                labels=[],
                split=split,
                task_type="dialogue",
                dataset="halueval",
                language="en",
            ))

        hallucinated_response = item.get("hallucinated_response", "")
        if hallucinated_response:
            samples.append(HallucinationSample(
                prompt=prompt,
                answer=hallucinated_response,
                labels=[{"start": 0, "end": len(hallucinated_response), "label": "hallucination"}],
                split=split,
                task_type="dialogue",
                dataset="halueval",
                language="en",
            ))

    return samples


def process_summarization_data(data: list[dict], split: str) -> list[HallucinationSample]:
    """Process HaluEval summarization data into HallucinationSamples."""
    samples = []
    for item in data:
        document = item.get("document", "")
        prompt = document

        right_summary = item.get("right_summary", "")
        if right_summary:
            samples.append(HallucinationSample(
                prompt=prompt,
                answer=right_summary,
                labels=[],
                split=split,
                task_type="summarization",
                dataset="halueval",
                language="en",
            ))

        hallucinated_summary = item.get("hallucinated_summary", "")
        if hallucinated_summary:
            samples.append(HallucinationSample(
                prompt=prompt,
                answer=hallucinated_summary,
                labels=[{"start": 0, "end": len(hallucinated_summary), "label": "hallucination"}],
                split=split,
                task_type="summarization",
                dataset="halueval",
                language="en",
            ))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Preprocess HaluEval data for LettuceDetect")
    parser.add_argument("--input_dir", type=str, default="data/halueval", help="Dir with HaluEval JSON files")
    parser.add_argument("--output_dir", type=str, default="data/halueval", help="Dir to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--max_samples_per_task", type=int, default=None, help="Cap samples per task type (for faster testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Process each HaluEval task type
    task_files = {
        "qa": ("qa_data.json", process_qa_data),
        "dialogue": ("dialogue_data.json", process_dialogue_data),
        "summarization": ("summarization_data.json", process_summarization_data),
    }

    for task_name, (filename, processor) in task_files.items():
        filepath = input_dir / filename
        if not filepath.exists():
            print(f"  Skipping {task_name}: {filepath} not found")
            continue

        print(f"Processing {task_name} data from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        # Cap samples if requested
        if args.max_samples_per_task and len(data) > args.max_samples_per_task:
            random.shuffle(data)
            data = data[:args.max_samples_per_task]

        # Split into train/test
        random.shuffle(data)
        split_idx = int(len(data) * args.train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        train_samples = processor(train_data, "train")
        test_samples = processor(test_data, "test")

        all_samples.extend(train_samples)
        all_samples.extend(test_samples)

        print(f"  {task_name}: {len(train_samples)} train + {len(test_samples)} test samples")

    # Create HallucinationData
    hallucination_data = HallucinationData(samples=all_samples)

    # Count stats
    train_count = sum(1 for s in all_samples if s.split == "train")
    test_count = sum(1 for s in all_samples if s.split == "test")
    hallucinated_count = sum(1 for s in all_samples if len(s.labels) > 0)
    supported_count = sum(1 for s in all_samples if len(s.labels) == 0)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  Train: {train_count}, Test: {test_count}")
    print(f"  Hallucinated: {hallucinated_count}, Supported: {supported_count}")

    # Save to JSON
    output_path = output_dir / "halueval_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hallucination_data.to_json(), f, indent=2)

    print(f"\nSaved processed data to {output_path}")


if __name__ == "__main__":
    main()
