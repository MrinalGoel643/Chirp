"""
scripts/model.py

Defines, trains, and evaluates all three models:
  1. NaiveBaseline    — majority-class DummyClassifier
  2. RandomForestModel — sklearn RandomForest on MFCC features
  3. EfficientNetModel — fine-tuned EfficientNet-B0 on mel spectrograms

Each model exposes a consistent .train() / .evaluate() / .predict() interface.

Usage (via setup.py — not called directly):
    from scripts.model import NaiveBaseline, RandomForestModel, EfficientNetModel

Attribution:
    timm — PyTorch Image Models — https://github.com/huggingface/pytorch-image-models
    EfficientNet: Tan & Le, 2019 — https://arxiv.org/abs/1905.11946
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CNN hyper-parameters ──────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
LR         = 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Naive Baseline
# ═══════════════════════════════════════════════════════════════════════════════

class NaiveBaseline:
    """
    Majority-class classifier. Always predicts the most frequent species
    in the training set regardless of input. Serves as the performance floor.
    """

    def __init__(self) -> None:
        self._clf = DummyClassifier(strategy="most_frequent", random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the dummy classifier.

        Args:
            X_train: Feature matrix (shape ignored — only y_train matters).
            y_train: Integer class labels.
        """
        self._clf.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return majority-class predictions for each sample in X."""
        return self._clf.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> dict:
        """
        Compute accuracy and macro F1 on the test set.

        Args:
            X_test:      Feature matrix.
            y_test:      True integer labels.
            class_names: List of species name strings.

        Returns:
            Dict with 'accuracy', 'macro_f1', and 'model' keys.
        """
        preds = self.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro", zero_division=0)
        print(f"[Naive Baseline]  Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        return {"model": "Naive Baseline", "accuracy": acc, "macro_f1": f1}

    def save(self, models_dir: Path) -> None:
        """Persist classifier to disk."""
        joblib.dump(self._clf, models_dir / "naive_baseline.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Random Forest
# ═══════════════════════════════════════════════════════════════════════════════

class RandomForestModel:
    """
    Random Forest classifier trained on MFCC feature vectors.

    MFCC features (mean + std over time) provide a compact, interpretable
    audio representation that works well with tree-based models even at
    small dataset sizes.
    """

    def __init__(self, n_estimators: int = 200) -> None:
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the Random Forest on MFCC features.

        Args:
            X_train: MFCC feature matrix of shape (N, n_mfcc * 2).
            y_train: Integer class labels.
        """
        print(f"Training Random Forest ({self._clf.n_estimators} trees)…")
        self._clf.fit(X_train, y_train)
        print("Done.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for each sample."""
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape (N, num_classes)."""
        return self._clf.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> dict:
        """
        Compute accuracy, macro F1, and a per-class classification report.

        Args:
            X_test:      MFCC feature matrix.
            y_test:      True integer labels.
            class_names: List of species name strings.

        Returns:
            Dict with 'accuracy', 'macro_f1', 'model', and 'preds' keys.
        """
        preds = self.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro", zero_division=0)
        print(f"[Random Forest]   Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        print(classification_report(y_test, preds, target_names=class_names, zero_division=0))
        return {"model": "Random Forest", "accuracy": acc, "macro_f1": f1, "preds": preds}

    def save(self, models_dir: Path) -> None:
        """Persist classifier to disk."""
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, models_dir / "random_forest.pkl")
        print(f"Saved → {models_dir / 'random_forest.pkl'}")

    @classmethod
    def load(cls, models_dir: Path) -> "RandomForestModel":
        """Load a previously saved Random Forest from disk."""
        instance = cls.__new__(cls)
        instance._clf = joblib.load(models_dir / "random_forest.pkl")
        return instance


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EfficientNet-B0 — dataset + model wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class MelSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that wraps pre-computed mel spectrogram arrays.

    Each 2-D mel array is normalised to [0, 255], stacked into 3 channels,
    and resized to IMG_SIZE × IMG_SIZE so it can be fed into EfficientNet-B0,
    which was pre-trained on ImageNet (3-channel, 224 × 224 images).
    """

    def __init__(self, X_mel: np.ndarray, y: np.ndarray, img_size: int = IMG_SIZE) -> None:
        self.X   = X_mel
        self.y   = y
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        mel = self.X[idx]  # (N_MELS, T)

        # Normalise to uint8 and broadcast to 3 channels
        mel_norm  = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        mel_uint8 = (mel_norm * 255).astype(np.uint8)
        img       = np.stack([mel_uint8] * 3, axis=-1)  # (H, W, 3)

        return self.transform(img), int(self.y[idx])


class EfficientNetModel:
    """
    EfficientNet-B0 fine-tuned end-to-end on mel spectrogram images.

    Pre-trained weights (ImageNet) are loaded via timm; the final
    classification head is replaced with a Linear(1280, num_classes) layer.
    Training uses AdamW + cosine annealing LR schedule with early stopping
    based on validation macro F1.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._model      = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        ).to(DEVICE)

    def _make_loaders(
        self,
        X_mel_train: np.ndarray, y_train: np.ndarray,
        X_mel_val:   np.ndarray, y_val:   np.ndarray,
        X_mel_test:  np.ndarray, y_test:  np.ndarray,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train / val / test DataLoaders."""
        train_ds = MelSpectrogramDataset(X_mel_train, y_train)
        val_ds   = MelSpectrogramDataset(X_mel_val,   y_val)
        test_ds  = MelSpectrogramDataset(X_mel_test,  y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Run one training epoch. Returns mean cross-entropy loss."""
        self._model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(self._model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader) -> tuple[float, float, np.ndarray]:
        """Evaluate on a DataLoader. Returns (accuracy, macro_f1, predictions)."""
        self._model.eval()
        all_preds, all_labels = [], []
        for imgs, labels in loader:
            preds = self._model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return acc, f1, np.array(all_preds)

    def train(
        self,
        X_mel_train: np.ndarray, y_train: np.ndarray,
        X_mel_val:   np.ndarray, y_val:   np.ndarray,
        X_mel_test:  np.ndarray, y_test:  np.ndarray,
        epochs: int = 20,
        models_dir: Path = Path("models"),
    ) -> dict:
        """
        Full training loop with validation-based model checkpointing.

        Args:
            X_mel_train / y_train: Training mel spectrograms and labels.
            X_mel_val   / y_val:   Validation mel spectrograms and labels.
            X_mel_test  / y_test:  Test mel spectrograms and labels.
            epochs:                Number of training epochs.
            models_dir:            Directory to save the best checkpoint.

        Returns:
            Dict with 'accuracy', 'macro_f1', 'history', 'model', and 'preds'.
        """
        train_loader, val_loader, test_loader = self._make_loaders(
            X_mel_train, y_train, X_mel_val, y_val, X_mel_test, y_test
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self._model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history     = {"train_loss": [], "val_acc": [], "val_f1": []}
        best_val_f1 = 0.0
        best_path   = models_dir / "efficientnet_best.pt"
        models_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss           = self._train_epoch(train_loader, optimizer, criterion)
            val_acc, val_f1, _   = self._eval_loader(val_loader)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(
                f"  Epoch [{epoch:02d}/{epochs}]  "
                f"Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}  Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self._model.state_dict(), best_path)

        # Load best checkpoint and evaluate on test set
        self._model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        test_acc, test_f1, preds = self._eval_loader(test_loader)
        print(f"[EfficientNet-B0]  Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

        return {
            "model":     "EfficientNet-B0",
            "accuracy":  test_acc,
            "macro_f1":  test_f1,
            "history":   history,
            "preds":     preds,
        }

    @torch.no_grad()
    def predict_proba(self, X_mel: np.ndarray) -> np.ndarray:
        """
        Return softmax class probabilities for a batch of mel spectrograms.

        Args:
            X_mel: Array of shape (N, N_MELS, T).

        Returns:
            Probability matrix of shape (N, num_classes).
        """
        self._model.eval()
        ds     = MelSpectrogramDataset(X_mel, np.zeros(len(X_mel), dtype=int))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        probs  = []
        for imgs, _ in loader:
            logits = self._model(imgs.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def save(self, models_dir: Path) -> None:
        """Save the current model weights (not necessarily the best checkpoint)."""
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), models_dir / "efficientnet_final.pt")

    @classmethod
    def load(cls, num_classes: int, models_dir: Path) -> "EfficientNetModel":
        """Load best checkpoint from disk for inference."""
        instance = cls(num_classes)
        state    = torch.load(models_dir / "efficientnet_best.pt", map_location=DEVICE)
        instance._model.load_state_dict(state)
        instance._model.eval()
        return instance
