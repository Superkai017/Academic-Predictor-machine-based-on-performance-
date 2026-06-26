"""
evaluation.py — Training evaluation, confusion matrix, and ROC curve utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from config import CLASS_NAMES, TARGET_COL


# ── Baseline comparison table ─────────────────────────────────────────────────

def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> None:
    """
    Fit every model in *models*, then print two summary tables:

    1. Train / test accuracy + over-fit flag.
    2. Correct / wrong / total counts per model.
    """
    # Fit all models first
    for model in models.values():
        model.fit(X_train, y_train)

    # ── Table 1: accuracy & over-fit ──────────────────────────────────────────
    print(f"\n{'Model':<6} | {'Train':>7} | {'Test':>7} | {'Gap':>7} | Status")
    print("-" * 50)
    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        gap       = train_acc - test_acc
        flag      = "OVERFIT" if gap > 0.1 else " OK"
        print(f"{name:<6} | {train_acc:>7.2%} | {test_acc:>7.2%} | {gap:>7.2%} | {flag}")

    # ── Table 2: counts ───────────────────────────────────────────────────────
    print(f"\n{'Model':<6} | {'Correct':>8} | {'Wrong':>8} | {'Total':>8} | {'Accuracy':>9}")
    print("-" * 50)
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        correct = (y_pred == y_test).sum()
        wrong   = (y_pred != y_test).sum()
        total   = len(y_test)
        print(
            f"{name:<6} | {correct:>8} | {wrong:>8} | {total:>8} "
            f"| {accuracy_score(y_test, y_pred):>9.2%}"
        )


# ── Feature correlation heatmap ───────────────────────────────────────────────

def plot_feature_correlation(X_sel_train, y_train, selected_features) -> None:
    """Heatmap of each selected feature's correlation with the target label."""
    df_sel = pd.DataFrame(X_sel_train, columns=selected_features)
    df_sel[TARGET_COL] = y_train.values

    plt.figure(figsize=(8, 6))
    corr = df_sel.corr()[[TARGET_COL]].drop(TARGET_COL)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation of Selected Features with Performance Tier", fontsize=16)
    plt.tight_layout()
    plt.show()


# ── Confusion matrix + ROC curve ──────────────────────────────────────────────

def plot_evaluation(best_model, X_test, y_test) -> None:
    """
    Side-by-side confusion matrix and one-vs-rest ROC curves for *best_model*.
    Also prints an AUC summary to stdout.
    """
    colors     = ["#F44336", "#FF9800", "#4CAF50"]
    y_pred     = best_model.predict(X_test)
    y_prob     = best_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Logistic Regression — Confusion Matrix & ROC Curve",
        fontsize=15, fontweight="bold",
    )

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100
            axes[0].text(j, i + 0.3, f"({pct:.1f}%)", ha="center", fontsize=9, color="gray")

    # ROC curves
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])  # type: ignore
        roc_auc     = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    axes[1].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    axes[1].fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    axes[1].set_title("ROC Curve (One-vs-Rest)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # AUC summary
    print("\n=== AUC Summary ===")
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])  # type: ignore
        print(f"  {name:10s} AUC : {auc(fpr, tpr):.4f}")
