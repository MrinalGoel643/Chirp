"""
app.py

FastAPI inference server for Warbler — bird audio species classifier.

Endpoints:
  POST /predict     — upload an audio file, returns top-K species predictions
  GET  /health      — liveness check
  GET  /classes     — list all supported species

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

The server loads the best model weights and label encoder from models/
on startup. Set HF_REPO_ID env var to pull weights from HuggingFace Hub
instead of local disk.
"""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from scripts.build_features import (
    AUDIO_DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    compute_mel_spectrogram,
    extract_mfcc,
    load_audio,
)
from scripts.model import EfficientNetModel

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
CONFIG_PATH = MODELS_DIR / "model_config.json"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Warbler — Bird Audio Classifier",
    description="Identify North American bird species from audio clips.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state (loaded once at startup) ────────────────────────────────
_model:   Optional[EfficientNetModel] = None
_le                                   = None
_config:  Optional[dict]              = None


def _load_from_hub(repo_id: str) -> None:
    """
    Download model artifacts from a HuggingFace Hub model repository.

    Args:
        repo_id: HuggingFace repo string, e.g. 'username/warbler-bird-classifier'.
    """
    from huggingface_hub import hf_hub_download

    files = ["efficientnet_best.pt", "label_encoder.pkl", "model_config.json"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in files:
        dest = MODELS_DIR / filename
        if not dest.exists():
            print(f"Downloading {filename} from {repo_id}…")
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            dest.write_bytes(Path(path).read_bytes())


@app.on_event("startup")
def load_model() -> None:
    """
    Load model weights, label encoder, and config at server startup.

    Pulls from HuggingFace Hub if HF_REPO_ID env var is set;
    otherwise loads from local models/ directory.
    """
    global _model, _le, _config

    hf_repo = os.getenv("HF_REPO_ID")
    if hf_repo:
        _load_from_hub(hf_repo)

    if not CONFIG_PATH.exists():
        raise RuntimeError(
            "model_config.json not found. Run setup.py or set HF_REPO_ID."
        )

    with open(CONFIG_PATH) as f:
        _config = json.load(f)

    _le    = joblib.load(MODELS_DIR / "label_encoder.pkl")
    _model = EfficientNetModel.load(num_classes=_config["num_classes"], models_dir=MODELS_DIR)

    print(f"Model loaded: {_config['best_model']}  |  {_config['num_classes']} classes")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _preprocess_audio(audio_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode uploaded audio bytes and extract MFCC + mel spectrogram features.

    Args:
        audio_bytes: Raw bytes of any librosa-supported audio format.

    Returns:
        Tuple of (mfcc_vector, mel_spectrogram) as numpy arrays.
    """
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio = load_audio(tmp_path, sr=SAMPLE_RATE, duration=AUDIO_DURATION)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    mfcc = extract_mfcc(audio)
    mel  = compute_mel_spectrogram(audio)
    return mfcc, mel


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Liveness check — returns model name and class count."""
    return {
        "status": "ok",
        "model":  _config["best_model"] if _config else "not loaded",
        "classes": _config["num_classes"] if _config else 0,
    }


@app.get("/classes")
def list_classes() -> dict:
    """Return all species codes the model can predict."""
    if _le is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"classes": _le.classes_.tolist()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    """
    Identify a bird species from an uploaded audio file.

    Args:
        file:  Audio file (.ogg, .mp3, .wav, .flac).
        top_k: Number of top predictions to return (default 3).

    Returns:
        JSON with top-K predictions, each containing:
          - species_code  : BirdCLEF species label (e.g. 'norcar')
          - confidence    : Softmax probability (0–1)
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        mfcc, mel = _preprocess_audio(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {exc}")

    # Run inference — mel shape must be (1, N_MELS, T)
    probs      = _model.predict_proba(mel[np.newaxis])  # (1, num_classes)
    probs_flat = probs[0]

    top_k      = min(top_k, len(_le.classes_))
    top_idx    = np.argsort(probs_flat)[::-1][:top_k]

    predictions = [
        {
            "species_code": _le.classes_[i],
            "confidence":   round(float(probs_flat[i]), 4),
        }
        for i in top_idx
    ]

    return {
        "predictions":  predictions,
        "model":        _config["best_model"],
        "top_species":  predictions[0]["species_code"],
        "confidence":   predictions[0]["confidence"],
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
