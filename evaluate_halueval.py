"""Comprehensive evaluation and performance analysis script for LettuceDetect on HaluEval.

Produces:
- Example-level metrics: F1, Precision, Recall, Accuracy
- Token-level metrics: F1, Precision, Recall  
- ROC curve plot saved as PNG
- Full classification report
- All results saved to JSON
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
)

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
)


def evaluate_token_level(model, dataloader, device):
    """Evaluate at token level - each token is classified as supported/hallucinated."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            mask = batch["labels"] != -100
            predictions = predictions[mask].cpu().numpy()
            labels = batch["labels"][mask].cpu().numpy()
            prob_class1 = probs[:, :, 1][mask].cpu().numpy()

            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(prob_class1.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    report = classification_report(
        all_labels, all_preds,
        target_names=["Supported", "Hallucinated"],
        digits=4, zero_division=0
    )

    return {
        "supported": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])},
        "hallucinated": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1])},
        "accuracy": float(acc),
        "auroc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def evaluate_example_level(model, dataloader, device):
    """Evaluate at example level - if any token is hallucinated, the whole example is hallucinated."""
    model.eval()
    example_preds = []
    example_labels = []
    example_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            for i in range(batch["labels"].size(0)):
                sample_labels = batch["labels"][i]
                sample_preds = predictions[i].cpu()
                valid_mask = sample_labels != -100

                if valid_mask.sum().item() == 0:
                    true_label = 0
                    pred_label = 0
                    max_prob = 0.0
                else:
                    sample_labels_valid = sample_labels[valid_mask].cpu()
                    sample_preds_valid = sample_preds[valid_mask]
                    sample_probs_valid = probs[i][valid_mask]

                    true_label = 1 if (sample_labels_valid == 1).any().item() else 0
                    pred_label = 1 if (sample_preds_valid == 1).any().item() else 0
                    max_prob = sample_probs_valid[:, 1].max().item()

                example_labels.append(true_label)
                example_preds.append(pred_label)
                example_probs.append(max_prob)

    precision, recall, f1, _ = precision_recall_fscore_support(
        example_labels, example_preds, labels=[0, 1], average=None, zero_division=0
    )
    acc = accuracy_score(example_labels, example_preds)
    cm = confusion_matrix(example_labels, example_preds, labels=[0, 1])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(example_labels, example_probs)
    roc_auc = auc(fpr, tpr)

    report = classification_report(
        example_labels, example_preds,
        target_names=["Supported", "Hallucinated"],
        digits=4, zero_division=0
    )

    return {
        "supported": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])},
        "hallucinated": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1])},
        "accuracy": float(acc),
        "auroc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def plot_roc_curves(token_results, example_results, output_path):
    """Plot ROC curves for both token-level and example-level evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Token-level ROC curve
    ax = axes[0]
    fpr = token_results["roc_data"]["fpr"]
    tpr = token_results["roc_data"]["tpr"]
    roc_auc = token_results["auroc"]
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Token-Level ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Example-level ROC curve
    ax = axes[1]
    fpr = example_results["roc_data"]["fpr"]
    tpr = example_results["roc_data"]["tpr"]
    roc_auc = example_results["auroc"]
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Example-Level ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle("LettuceDetect on HaluEval - ROC Curves", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nROC curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LettuceDetect on HaluEval data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to halueval_data.json")
    parser.add_argument("--output_dir", type=str, default="output/evaluation", help="Dir to save results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = Path(args.data_path)
    hallucination_data = HallucinationData.from_json(json.loads(data_path.read_text()))
    test_samples = [s for s in hallucination_data.samples if s.split == "test"]

    print(f"Loaded {len(test_samples)} test samples")
    print(f"  Hallucinated: {sum(1 for s in test_samples if len(s.labels) > 0)}")
    print(f"  Supported: {sum(1 for s in test_samples if len(s.labels) == 0)}")

    # Group by task type
    task_types = {}
    for s in test_samples:
        task_types.setdefault(s.task_type, []).append(s)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    # Evaluate on full test set
    test_dataset = HallucinationDataset(test_samples, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    print("\n" + "=" * 60)
    print("TOKEN-LEVEL EVALUATION (Full Test Set)")
    print("=" * 60)
    token_results = evaluate_token_level(model, test_loader, device)
    print(token_results["classification_report"])
    print(f"Accuracy: {token_results['accuracy']:.4f}")
    print(f"AUROC: {token_results['auroc']:.4f}")

    print("\n" + "=" * 60)
    print("EXAMPLE-LEVEL EVALUATION (Full Test Set)")
    print("=" * 60)
    example_results = evaluate_example_level(model, test_loader, device)
    print(example_results["classification_report"])
    print(f"Accuracy: {example_results['accuracy']:.4f}")
    print(f"AUROC: {example_results['auroc']:.4f}")

    # Per-task evaluation
    per_task_results = {}
    for task_type, samples in task_types.items():
        print(f"\n{'=' * 60}")
        print(f"EXAMPLE-LEVEL EVALUATION - Task: {task_type} ({len(samples)} samples)")
        print(f"{'=' * 60}")
        task_dataset = HallucinationDataset(samples, tokenizer)
        task_loader = DataLoader(task_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
        task_results = evaluate_example_level(model, task_loader, device)
        print(task_results["classification_report"])
        print(f"Accuracy: {task_results['accuracy']:.4f}")
        print(f"AUROC: {task_results['auroc']:.4f}")
        per_task_results[task_type] = task_results

    # Plot ROC curves
    roc_path = output_dir / "roc_curves.png"
    plot_roc_curves(token_results, example_results, roc_path)

    # Save all results
    all_results = {
        "token_level": {k: v for k, v in token_results.items() if k != "roc_data"},
        "example_level": {k: v for k, v in example_results.items() if k != "roc_data"},
        "per_task": {
            task: {k: v for k, v in res.items() if k != "roc_data"}
            for task, res in per_task_results.items()
        },
    }
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Token-Level':>12} {'Example-Level':>14}")
    print("-" * 55)
    print(f"{'Hallucinated F1':<25} {token_results['hallucinated']['f1']:>12.4f} {example_results['hallucinated']['f1']:>14.4f}")
    print(f"{'Hallucinated Precision':<25} {token_results['hallucinated']['precision']:>12.4f} {example_results['hallucinated']['precision']:>14.4f}")
    print(f"{'Hallucinated Recall':<25} {token_results['hallucinated']['recall']:>12.4f} {example_results['hallucinated']['recall']:>14.4f}")
    print(f"{'Supported F1':<25} {token_results['supported']['f1']:>12.4f} {example_results['supported']['f1']:>14.4f}")
    print(f"{'Accuracy':<25} {token_results['accuracy']:>12.4f} {example_results['accuracy']:>14.4f}")
    print(f"{'AUROC':<25} {token_results['auroc']:>12.4f} {example_results['auroc']:>14.4f}")


if __name__ == "__main__":
    main()
