import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


THRESHOLDS = None  # set by train/evaluate at import time if needed


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.array(labels)

    # Sigmoid probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Class-wise thresholds if provided, else default 0.5
    if THRESHOLDS is not None:
        thr = np.array(THRESHOLDS, dtype=float)
        if thr.ndim == 1:
            thr = thr.reshape(1, -1)
        preds = (probs >= thr).astype(int)
    else:
        preds = (probs >= 0.5).astype(int)

    # Example-based (subset) accuracy
    subset_acc = (preds == labels).all(axis=1).mean()

    # Micro metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels.flatten(), preds.flatten(), average="micro", zero_division=0
    )

    # Per-class metrics
    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Expose per-class metrics with prefixes
    metrics = {
        "accuracy": subset_acc,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

    for i, (p, r, f, s) in enumerate(zip(precision_c, recall_c, f1_c, support_c)):
        metrics[f"precision_{i}"] = p
        metrics[f"recall_{i}"] = r
        metrics[f"f1_{i}"] = f
        metrics[f"support_{i}"] = s

    return metrics
