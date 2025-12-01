"""Model evaluation metrics and visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> dict:
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def get_classification_report(y_true, y_pred) -> str:
    """Generate classification report."""
    return classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])


def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    return fig, ax


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 6)):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()

    return fig, ax
