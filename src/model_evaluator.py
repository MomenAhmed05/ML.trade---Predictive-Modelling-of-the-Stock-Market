"""
Statistical Evaluation and Visualization Pipeline

This module provides a comprehensive suite for evaluating both discrete (classification) 
and continuous (regression) outputs from the multi-task LSTM model. 

It standardizes the generation of academic-grade performance metrics and visual diagnostics, 
ensuring the model's predictive validity can be empirically verified.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error,
)


class ModelEvaluator:
    """
    Centralized handler for all statistical metric computations and matplotlib visualizations.
    Automatically persists generated graphs to a centralized results directory for reporting.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Computes standard binary classification metrics to assess the model's 
        ability to correctly forecast directional market movement (UP/DOWN).
        
        Zero-division guards are applied to prevent runtime faults during early training epochs 
        where the model may collapse into predicting only a single majority class.
        """
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        print("\n" + "=" * 40)
        print("CLASSIFICATION METRICS")
        print("=" * 40)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Computes continuous error metrics to assess the accuracy of the absolute price forecast.
        
        Args:
            y_true: Ground truth actuals (normalized [0,1] space)
            y_pred: Model predictions (normalized [0,1] space)
        """
        # Enforce 1D structure to prevent internal broadcasting mismatches during array operations
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print("\n" + "=" * 40)
        print("REGRESSION METRICS")
        print("=" * 40)
        print(f"MSE:  {mse:.6f}")
        print(f"MAE:  {mae:.6f}")
        
        return {"mse": mse, "mae": mae}

    def plot_training_history(self, history, save_path: str = "training_history.png") -> None:
        """
        Generates a 1x2 grid visualizing the progression of loss functions and accuracy across 
        training epochs for the directional classification task.
        
        Divergence between the Training (blue) and Validation (orange) curves provides visual 
        evidence of overfitting dynamics.
        """
        # Extract dictionary payload from Keras History object
        h = history.history if hasattr(history, "history") else history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Classification Loss (Focal/Cross-Entropy)
        axes[0].plot(h.get("loss", []), label="Train", linewidth=2)
        axes[0].plot(h.get("val_loss", []), label="Validation", linewidth=2)
        axes[0].set_title("Classification Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Classification Accuracy
        axes[1].plot(h.get("accuracy", []), label="Train", linewidth=2)
        axes[1].plot(h.get("val_accuracy", []), label="Validation", linewidth=2)
        axes[1].set_title("Direction Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        full_path = self.results_dir / save_path
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training history plot saved to: {full_path}")

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "confusion_matrix.png"
    ) -> None:
        """
        Visualizes classification performance mapping true directional states against model predictions.
        Crucial for detecting if the model has developed an underlying bias toward predicting 
        only UP or only DOWN scenarios.
        """
        cm = confusion_matrix(y_true, y_pred)
        labels = ["DOWN", "UP"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Direction")
        ax.set_ylabel("Actual Direction")
        ax.set_title("Confusion Matrix")

        # Dynamically contrast text colors over the heatmap to ensure readability
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        full_path = self.results_dir / save_path
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to: {full_path}")

    def plot_predictions_vs_actual(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "predictions_vs_actual.png"
    ) -> None:
        """
        Overlays the predicted regression timeline against the ground truth.
        (Usually constrained to a small subset of the data array to preserve visual clarity).
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual Normalized Price", alpha=0.8, color="blue", linewidth=2)
        plt.plot(y_pred, label="Predicted Normalized Price", alpha=0.8, color="orange", linestyle="--", linewidth=2)
        plt.title("Predictions vs Actual Prices (Subset)")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        full_path = self.results_dir / save_path
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"Predictions vs actual plot saved to: {full_path}")
