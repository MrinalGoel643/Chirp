"""
setup.py

End-to-end pipeline script for Warbler — bird audio species classifier.

Runs in order:
  1. Load pre-extracted features (or run build_features.py first)
  2. Train / val / test split
  3. Train all three models (Naive Baseline, Random Forest, EfficientNet-B0)
  4. Evaluate and compare
  5. Save best model + config for app.py

Usage:
    # First time (download + feature extraction):
    python scripts/make_dataset.py
    python scripts/build_features.py
    python setup.py

    # If features already exist:
    python setup.py --epochs 20
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.model import EfficientNetModel, NaiveBaseline, RandomForestModel


# ── Defaults ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUTS_DIR   = Path("data/outputs")
SEED          = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.10


def load_features(processed_dir: Path) -> tuple:
    """
    Load pre-computed feature arrays and label encoder from disk.

    Args:
        processed_dir: Directory containing .npy files and label_encoder.pkl.

    Returns:
        Tuple of (X_mfcc, X_mel, y, label_encoder).

    Raises:
        FileNotFoundError: If feature files are missing — run build_features.py first.
    """
    required = ["X_mfcc.npy", "X_mel.npy", "y.npy", "label_encoder.pkl"]
    for f in required:
        if not (processed_dir / f).exists():
            raise FileNotFoundError(
                f"Missing {f} in {processed_dir}. "
                "Run `python scripts/build_features.py` first."
            )

    X_mfcc = np.load(processed_dir / "X_mfcc.npy")
    X_mel  = np.load(processed_dir / "X_mel.npy")
    y      = np.load(processed_dir / "y.npy")
    le     = joblib.load(processed_dir / "label_encoder.pkl")

    print(f"Loaded features: {len(y)} samples, {len(le.classes_)} classes")
    print(f"  X_mfcc: {X_mfcc.shape}  X_mel: {X_mel.shape}")
    return X_mfcc, X_mel, y, le


def make_splits(
    X_mfcc: np.ndarray,
    X_mel:  np.ndarray,
    y:      np.ndarray,
    test_size: float = TEST_SIZE,
    val_size:  float = VAL_SIZE,
    seed: int = SEED,
) -> tuple[dict, dict, dict]:
    """
    Create stratified train / val / test splits.

    Args:
        X_mfcc:    MFCC feature matrix.
        X_mel:     Mel spectrogram array.
        y:         Integer label array.
        test_size: Fraction of data for the test set.
        val_size:  Fraction of data for the validation set.
        seed:      Random seed for reproducibility.

    Returns:
        Three dicts each with keys 'mfcc', 'mel', 'y'.
    """
    idx = np.arange(len(y))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    val_frac = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, stratify=y[idx_trainval], random_state=seed
    )

    def subset(idx_):
        return {"mfcc": X_mfcc[idx_], "mel": X_mel[idx_], "y": y[idx_]}

    train, val, test = subset(idx_train), subset(idx_val), subset(idx_test)
    print(f"Split — Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")
    return train, val, test


def save_results(results: list[dict], outputs_dir: Path) -> None:
    """
    Save model comparison table as CSV and print a summary.

    Args:
        results:     List of result dicts from each model's .evaluate() call.
        outputs_dir: Directory to write model_comparison.csv.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"Model": r["model"], "Test Accuracy": r["accuracy"], "Macro F1": r["macro_f1"]}
        for r in results
    ])
    df.to_csv(outputs_dir / "model_comparison.csv", index=False)

    print("\n=== MODEL COMPARISON ===")
    print(df.to_string(index=False))


def save_model_config(best: dict, le, models_dir: Path) -> None:
    """
    Persist the model config JSON consumed by app.py at startup.

    Args:
        best:       Result dict of the winning model.
        le:         Fitted LabelEncoder.
        models_dir: Directory to write model_config.json.
    """
    config = {
        "best_model":      best["model"],
        "test_accuracy":   round(best["accuracy"], 4),
        "test_macro_f1":   round(best["macro_f1"],  4),
        "classes":         le.classes_.tolist(),
        "num_classes":     len(le.classes_),
        "sample_rate":     22050,
        "audio_duration":  5,
        "n_mels":          128,
        "n_fft":           2048,
        "hop_length":      512,
        "n_mfcc":          40,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nBest model: {best['model']}  (Macro F1: {best['macro_f1']:.4f})")
    print(f"Config saved → {models_dir / 'model_config.json'}")


def run_pipeline(epochs: int = 20) -> None:
    """
    Execute the full training pipeline.

    Args:
        epochs: Number of epochs for EfficientNet-B0 training.
    """
    # ── 1. Load features ──────────────────────────────────────────────────────
    X_mfcc, X_mel, y, le = load_features(PROCESSED_DIR)
    num_classes           = len(le.classes_)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    train, val, test = make_splits(X_mfcc, X_mel, y)

    results = []

    # ── 3a. Naive Baseline ────────────────────────────────────────────────────
    print("\n── Naive Baseline ──")
    nb = NaiveBaseline()
    nb.train(train["mfcc"], train["y"])
    results.append(nb.evaluate(test["mfcc"], test["y"], le.classes_.tolist()))
    nb.save(MODELS_DIR)

    # ── 3b. Random Forest ─────────────────────────────────────────────────────
    print("\n── Random Forest ──")
    rf = RandomForestModel(n_estimators=200)
    rf.train(train["mfcc"], train["y"])
    results.append(rf.evaluate(test["mfcc"], test["y"], le.classes_.tolist()))
    rf.save(MODELS_DIR)

    # ── 3c. EfficientNet-B0 ───────────────────────────────────────────────────
    print(f"\n── EfficientNet-B0 ({epochs} epochs) ──")
    cnn = EfficientNetModel(num_classes=num_classes)
    cnn_result = cnn.train(
        train["mel"], train["y"],
        val["mel"],   val["y"],
        test["mel"],  test["y"],
        epochs=epochs,
        models_dir=MODELS_DIR,
    )
    results.append(cnn_result)

    # Also save label encoder alongside model weights
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    # ── 4. Compare & save ─────────────────────────────────────────────────────
    save_results(results, OUTPUTS_DIR)
    best = max(results, key=lambda r: r["macro_f1"])
    save_model_config(best, le, MODELS_DIR)

    print("\n✅ Pipeline complete. Artifacts in models/ and data/outputs/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Warbler bird classifier.")
    parser.add_argument("--epochs", type=int, default=20, help="EfficientNet training epochs")
    args = parser.parse_args()
    run_pipeline(epochs=args.epochs)


if __name__ == "__main__":
    main()
