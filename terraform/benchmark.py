import argparse
import json
import os
import platform
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET_COLUMN = "Class"


def load_dataset(data_path: str):
    """Load the Credit Card Fraud dataset and measure loading time."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    start_time = time.perf_counter()
    df = pd.read_csv(path)
    load_time = time.perf_counter() - start_time

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return df, X, y, load_time


def train_model(X_train, y_train):
    """Train LightGBM model and measure training time."""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    return model, training_time


def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def benchmark_inference(model, X_test):
    """
    Measure inference latency for 1 row
    and throughput for 1000 rows.
    """

    # ---------------------------------------------------------
    # Inference latency: exactly 1 row
    # ---------------------------------------------------------
    one_row = X_test.iloc[[0]]

    start_time = time.perf_counter()
    model.predict(one_row)
    latency_seconds = time.perf_counter() - start_time

    latency_ms = latency_seconds * 1000

    # ---------------------------------------------------------
    # Inference throughput: exactly 1000 rows
    # ---------------------------------------------------------
    benchmark_rows = min(1000, len(X_test))
    batch = X_test.iloc[:benchmark_rows]

    start_time = time.perf_counter()
    model.predict(batch)
    inference_seconds = time.perf_counter() - start_time

    throughput = benchmark_rows / inference_seconds

    return {
        "latency_1_row_ms": latency_ms,
        "throughput_1000_rows_samples_per_sec": throughput,
        "throughput_rows": benchmark_rows,
        "throughput_time_seconds": inference_seconds,
    }


def save_results(results, output_path: str):
    """Save benchmark results to JSON."""
    output = Path(output_path)

    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nResults saved to: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Day16 Track 2 LightGBM fraud detection benchmark"
    )

    parser.add_argument(
        "--data",
        default="ml-benchmark/creditcard.csv",
        help="Path to creditcard.csv",
    )

    parser.add_argument(
        "--output",
        default="benchmark_result.json",
        help="Path for benchmark result JSON",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Day16 Track 2 - LightGBM Benchmark")
    print("=" * 60)

    # =========================================================
    # 1. Load dataset
    # =========================================================
    print("\n[1/5] Loading dataset...")

    df, X, y, load_time = load_dataset(args.data)

    print(f"Dataset shape: {df.shape}")
    print(f"Samples: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fraud samples: {int(y.sum())}")
    print(f"Fraud ratio: {y.mean():.6%}")
    print(f"Data loading time: {load_time:.4f} seconds")

    # =========================================================
    # 2. Train/test split
    # =========================================================
    print("\n[2/5] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # =========================================================
    # 3. Train model
    # =========================================================
    print("\n[3/5] Training LightGBM...")

    model, training_time = train_model(X_train, y_train)

    print(f"Training time: {training_time:.4f} seconds")

    # =========================================================
    # 4. Evaluate model
    # =========================================================
    print("\n[4/5] Evaluating model...")

    metrics = evaluate_model(model, X_test, y_test)

    print(f"AUC-ROC:   {metrics['auc_roc']:.6f}")
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"F1:        {metrics['f1']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")

    # =========================================================
    # 5. Benchmark inference
    # =========================================================
    print("\n[5/5] Benchmarking inference...")

    inference = benchmark_inference(model, X_test)

    print(
        f"Inference latency (1 row): "
        f"{inference['latency_1_row_ms']:.4f} ms"
    )

    print(
        f"Inference throughput (1000 rows): "
        f"{inference['throughput_1000_rows_samples_per_sec']:.2f} "
        f"samples/sec"
    )

    # =========================================================
    # Build benchmark result
    # =========================================================
    results = {
        "dataset": {
            "path": str(Path(args.data).resolve()),
            "samples": int(len(df)),
            "features": int(X.shape[1]),
            "fraud_samples": int(y.sum()),
            "fraud_ratio": float(y.mean()),
        },
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "model": {
            "name": "LightGBM LGBMClassifier",
            "parameters": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
        },
        "benchmark": {
            "data_load_seconds": float(load_time),
            "training_seconds": float(training_time),
            "inference_latency_1_row_ms": float(
                inference["latency_1_row_ms"]
            ),
            "inference_throughput_1000_rows_samples_per_sec": float(
                inference["throughput_1000_rows_samples_per_sec"]
            ),
        },
        "metrics": {
            "auc_roc": float(metrics["auc_roc"]),
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": int(os.cpu_count() or 0),
        },
    }

    save_results(results, args.output)

    print("\n" + "=" * 60)
    print("Benchmark completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
