#!/usr/bin/env python
"""
Train Logistic Regression model for accept probability prediction.

Usage:
    python litaagent_os/scripts/train_logistic.py \
        --dataset assets/accept_dataset.parquet \
        --output assets/models/accept
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Features to use for training (must match LinearLogisticAcceptModel)
FEATURE_COLS = [
    "round_rel",      # 相对轮次 [0, 1]
    "role_is_seller", # 是否为卖方 {0, 1}
    "need_norm",      # 归一化需求 [0, 1] (remain / total)
    "q_norm",         # 归一化数量 [0, 1]
    "p_bin",          # 价格档位 {0, 1}
    "p_norm",         # 归一化价格 [0, 1]
]


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load and validate dataset."""
    df = pd.read_parquet(dataset_path)
    
    # Validate required columns
    missing_cols = set(FEATURE_COLS + ["y_accept"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    print(f"Loaded dataset: {len(df)} samples")
    print(f"  y_accept=1: {df['y_accept'].sum()} ({100*df['y_accept'].mean():.1f}%)")
    print(f"  y_accept=0: {(1-df['y_accept']).sum()} ({100*(1-df['y_accept'].mean()):.1f}%)")
    
    return df


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                class_weight: str = "balanced") -> LogisticRegression:
    """Train logistic regression model."""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray, 
                   split_name: str = "Test") -> dict:
    """Evaluate model and return metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "log_loss": log_loss(y, y_pred_proba),
        "brier_score": brier_score_loss(y, y_pred_proba),
        "roc_auc": roc_auc_score(y, y_pred_proba),
    }
    
    print(f"\n{split_name} Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    
    return metrics


def plot_calibration_curve(model: LogisticRegression, X: np.ndarray, y: np.ndarray,
                           output_path: Path):
    """Plot calibration curve and save to file."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy="uniform")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", label="Logistic Regression")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nCalibration curve saved to: {output_path}")


def save_model_weights(model: LogisticRegression, feature_names: list, 
                       output_dir: Path, metrics: dict):
    """Save model weights in JSON format for LinearLogisticAcceptModel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract weights - use format expected by LinearLogisticAcceptModel.load()
    weights = {
        "bias": float(model.intercept_[0]),
        "weights": {
            name: float(coef) 
            for name, coef in zip(feature_names, model.coef_[0])
        },
        # Store feature order for verification
        "feature_order": feature_names,
    }
    
    # Save model.bin (JSON format)
    model_bin_path = output_dir / "model.bin"
    with open(model_bin_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"\nModel weights saved to: {model_bin_path}")
    
    # Save model_meta.json
    meta = {
        "model_type": "logistic",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "features": feature_names,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "class_weight": "balanced",
    }
    meta_path = output_dir / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Model metadata saved to: {meta_path}")
    
    return weights


def print_feature_importance(model: LogisticRegression, feature_names: list):
    """Print feature importance (coefficient magnitudes)."""
    print("\nFeature Coefficients:")
    print(f"  {'Feature':<20} {'Coefficient':>12}")
    print("  " + "-" * 34)
    for name, coef in sorted(zip(feature_names, model.coef_[0]), 
                              key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<20} {coef:>12.4f}")
    print(f"  {'(intercept)':<20} {model.intercept_[0]:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression for accept prediction")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to accept_dataset.parquet")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for model files")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    parser.add_argument("--class-weight", type=str, default="balanced",
                        choices=["balanced", "none"],
                        help="Class weight strategy (default: balanced)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip calibration plot")
    args = parser.parse_args()
    
    # Load data
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    df = load_dataset(args.dataset)
    
    # Prepare features and target
    X = df[FEATURE_COLS].values
    y = df["y_accept"].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain/Test split: {len(X_train)} / {len(X_test)}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)
    class_weight = None if args.class_weight == "none" else args.class_weight
    model = train_model(X_train, y_train, class_weight=class_weight)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Feature importance
    print_feature_importance(model, FEATURE_COLS)
    
    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    output_dir = Path(args.output)
    weights = save_model_weights(model, FEATURE_COLS, output_dir, test_metrics)
    
    # Calibration plot
    if not args.no_plot:
        try:
            plot_calibration_curve(model, X_test, y_test, 
                                   output_dir / "calibration_curve.png")
        except Exception as e:
            print(f"\nWarning: Could not create calibration plot: {e}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel files in: {output_dir}")
    print("  - model.bin         (weights in JSON)")
    print("  - model_meta.json   (metadata)")
    if not args.no_plot:
        print("  - calibration_curve.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
