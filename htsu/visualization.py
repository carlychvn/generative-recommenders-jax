"""Visualization utilities for HTSU models.

Provides functions to visualize:
- Training curves
- Attention patterns
- Confusion matrices
- Embedding spaces
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def load_training_metrics(metrics_file: Path) -> List[Dict]:
    """Load metrics from JSONL file.

    Args:
        metrics_file: Path to metrics.jsonl file

    Returns:
        List of metric dictionaries
    """
    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def print_training_summary(metrics: List[Dict]):
    """Print a summary of training metrics.

    Args:
        metrics: List of metric dictionaries from training
    """
    if not metrics:
        print("No metrics to display")
        return

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)

    # Get initial and final metrics
    initial = metrics[0]
    final = metrics[-1]

    print(f"Total steps: {final['step']}")
    print(f"Total time: {final['elapsed_time']:.1f}s")
    print(f"Steps/second: {final['step'] / final['elapsed_time']:.2f}")

    print("\nMetrics:")
    metric_keys = [k for k in initial.keys() if k not in ['step', 'elapsed_time']]

    for key in metric_keys:
        if key in initial and key in final:
            initial_val = initial[key]
            final_val = final[key]
            change = ((final_val - initial_val) / initial_val * 100) if initial_val != 0 else 0
            print(f"  {key:20s}: {initial_val:.4f} -> {final_val:.4f} ({change:+.1f}%)")

    print("=" * 80 + "\n")


def print_ascii_plot(
    values: List[float],
    width: int = 80,
    height: int = 20,
    title: Optional[str] = None
):
    """Print an ASCII plot of values.

    Args:
        values: List of values to plot
        width: Width of the plot in characters
        height: Height of the plot in characters
        title: Optional title for the plot
    """
    if not values:
        print("No values to plot")
        return

    values = np.array(values)

    if title:
        print("\n" + title)
        print("-" * width)

    # Normalize values to plot height
    min_val = np.min(values)
    max_val = np.max(values)
    value_range = max_val - min_val if max_val != min_val else 1

    # Create canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    # Downsample values to fit width
    indices = np.linspace(0, len(values) - 1, width, dtype=int)
    plot_values = values[indices]

    # Plot values
    for x, val in enumerate(plot_values):
        # Normalize to height
        y = int((val - min_val) / value_range * (height - 1))
        y = height - 1 - y  # Flip y-axis
        if 0 <= y < height:
            canvas[y][x] = '*'

    # Print canvas
    for row in canvas:
        print(''.join(row))

    # Print axis labels
    print("-" * width)
    print(f"Min: {min_val:.4f}  Max: {max_val:.4f}  Final: {values[-1]:.4f}")
    print()


def visualize_training_log(log_dir: Path):
    """Load and visualize training logs.

    Args:
        log_dir: Directory containing metrics.jsonl
    """
    metrics_file = log_dir / "metrics.jsonl"

    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return

    # Load metrics
    metrics = load_training_metrics(metrics_file)

    # Print summary
    print_training_summary(metrics)

    # Plot loss curve
    if metrics and 'loss' in metrics[0]:
        losses = [m['loss'] for m in metrics]
        print_ascii_plot(losses, title="Training Loss")

    # Plot other metrics
    if metrics:
        metric_keys = [k for k in metrics[0].keys()
                      if k not in ['step', 'elapsed_time', 'loss']]

        for key in metric_keys:
            if all(key in m for m in metrics):
                values = [m[key] for m in metrics]
                print_ascii_plot(values, title=f"{key.upper()}")


def print_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
):
    """Print a formatted confusion matrix.

    Args:
        confusion_matrix: (num_classes, num_classes) array
        class_names: Optional list of class names
    """
    num_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Calculate column widths
    max_name_len = max(len(name) for name in class_names)
    cell_width = max(8, max_name_len + 2)

    print("\n" + "=" * 80)
    print("Confusion Matrix")
    print("=" * 80)

    # Header
    print(" " * cell_width, end="")
    print("Predicted")
    print(" " * cell_width, end="")
    for name in class_names:
        print(f"{name:^{cell_width}}", end="")
    print()

    # Rows
    print("Actual")
    for i, name in enumerate(class_names):
        print(f"{name:>{cell_width}}", end="")
        for j in range(num_classes):
            print(f"{confusion_matrix[i, j]:^{cell_width}}", end="")
        print()

    print("=" * 80 + "\n")

    # Print per-class metrics
    print("Per-Class Metrics:")
    for i, name in enumerate(class_names):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  {name:20s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    print()


def print_metric_comparison(results: Dict[str, float], title: str = "Model Comparison"):
    """Print a formatted comparison of metrics across models.

    Args:
        results: Dictionary mapping model names to metric values
        title: Title for the comparison table
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    max_name_len = max(len(name) for name in results.keys())

    # Sort by metric value (ascending for losses, descending for accuracies)
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    for i, (name, value) in enumerate(sorted_results):
        rank_symbol = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        print(f"{rank_symbol} {name:{max_name_len}s}: {value:.4f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    """Example usage for visualization utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize HTSU training logs")
    parser.add_argument("log_dir", type=Path, help="Path to log directory")

    args = parser.parse_args()

    visualize_training_log(args.log_dir)
