"""
The evaluation measures, in one place so every step scores the same way.

Gender: accuracy, F1 (male = positive class), ROC-AUC, ECE.
Age: accuracy, +-1-bin accuracy, macro-F1, weighted-F1, balanced accuracy, one-vs-rest macro AUC, ECE.
"""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def ece(probabilities, labels, bins=10):
    """Expected calibration error with equal-width confidence bins.

    probabilities: (n, C) class probabilities, or (n,) probability of class 1 for a binary task.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == np.asarray(labels)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    # a confidence of exactly 1.0 belongs to the last bin
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        inside = which == b
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return total


def gender_scores(p_male, y_male):
    p_male = np.asarray(p_male, dtype=float)
    predicted = (p_male >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_male, predicted),
            "f1": f1_score(y_male, predicted),
            "auc": roc_auc_score(y_male, p_male),
            "ece": ece(p_male, y_male)}


def ovr_auc(probabilities, labels, n_classes):
    """One-vs-rest macro AUC over the classes present in `labels` (a class absent from a test split
    has no AUC)."""
    labels = np.asarray(labels)
    aucs = []
    for c in range(n_classes):
        positive = labels == c
        if positive.any() and not positive.all():
            aucs.append(roc_auc_score(positive, probabilities[:, c]))
    return float(np.mean(aucs))


def age_scores(probabilities, labels, tolerance=1):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    predicted = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    scores = {"accuracy": accuracy_score(labels, predicted),
              "within_1": float(np.mean(np.abs(predicted - labels) <= 1)),
              "macro_f1": f1_score(labels, predicted, average="macro", labels=range(n_classes), zero_division=0),
              "weighted_f1": f1_score(labels, predicted, average="weighted", zero_division=0),
              "balanced": balanced_accuracy_score(labels, predicted),
              "auc_ovr": ovr_auc(probabilities, labels, n_classes),
              "ece": ece(probabilities, labels)}
    if tolerance != 1:
        scores[f"within_{tolerance}"] = float(np.mean(np.abs(predicted - labels) <= tolerance))
    return scores
