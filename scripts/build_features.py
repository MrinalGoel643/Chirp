"""
scripts/build_features.py

Extracts MFCC feature vectors (for Random Forest) and log-scaled mel
spectrograms (for EfficientNet-B0) from all audio clips in the filtered
metadata. Saves results as .npy arrays and a fitted LabelEncoder.

Usage:
    python scripts/build_features.py

Attribution:
    librosa audio analysis library — https://librosa.org
"""

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ── Audio / feature constants ──────────────────────────────────────────────────
SAMPLE_RATE    = 22050
AUDIO_DURATION = 5       # seconds — clips are trimmed or padded to this length
N_MFCC         = 40      # number of MFCC coefficients
N_MELS         = 128     # mel frequency bins
N_FFT          = 2048
HOP_LENGTH     = 512


def load_audio(filepath: str, sr: int = SAMPLE_RATE, duration: int = AUDIO_DURATION) -> np.ndarray:
    """
    Load an audio file and pad or trim to a fixed duration.

    Args:
        filepath: Path to the .ogg / .mp3 / .wav file.
        sr:       Target sample rate (Hz).
        duration: Desired clip length in seconds.

    Returns:
        1-D float32 numpy array of shape (sr * duration,).
    """
    target_len = sr * duration
    audio, _   = librosa.load(filepath, sr=sr, duration=duration, mono=True)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")

    return audio[:target_len].astype(np.float32)


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Compute a fixed-length MFCC feature vector via mean and std pooling.

    Args:
        audio:  1-D audio signal.
        sr:     Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Feature vector of shape (n_mfcc * 2,)  — [mean | std].
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute a log-scaled (dB) mel spectrogram suitable as CNN input.

    Args:
        audio:      1-D audio signal.
        sr:         Sample rate.
        n_mels:     Number of mel filter banks.
        n_fft:      FFT window size.
        hop_length: Hop size between frames.

    Returns:
        2-D float32 array of shape (n_mels, time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def build_feature_arrays(
    df: pd.DataFrame,
    audio_root: Path,
    audio_subdir: str = "train_audio",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over all rows in the metadata DataFrame and extract features.

    Args:
        df:           Filtered metadata with 'filename' and 'primary_label' columns.
        audio_root:   Root data directory (data/raw/).
        audio_subdir: Subdirectory containing species sub-folders of .ogg files.

    Returns:
        Tuple of (X_mfcc, X_mel, labels) as numpy arrays.
        X_mfcc shape: (N, N_MFCC * 2)
        X_mel  shape: (N, N_MELS, time_frames)
        labels shape: (N,)  — string species codes
    """
    X_mfcc, X_mel, labels = [], [], []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        filepath = audio_root / audio_subdir / row["filename"]
        try:
            audio = load_audio(str(filepath))
            X_mfcc.append(extract_mfcc(audio))
            X_mel.append(compute_mel_spectrogram(audio))
            labels.append(row["primary_label"])
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"  Warning — failed to load {filepath}: {exc}")

    print(f"Feature extraction complete. Failed: {failed}/{len(df)}")
    return np.array(X_mfcc), np.array(X_mel), np.array(labels)


def save_features(
    X_mfcc: np.ndarray,
    X_mel: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    processed_dir: Path,
) -> None:
    """
    Persist feature arrays, encoded labels, and the LabelEncoder to disk.

    Args:
        X_mfcc:        MFCC feature matrix.
        X_mel:         Mel spectrogram array.
        y:             Integer-encoded label array.
        le:            Fitted LabelEncoder (needed to decode predictions later).
        processed_dir: Directory to save .npy files and encoder.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "X_mfcc.npy",  X_mfcc)
    np.save(processed_dir / "X_mel.npy",   X_mel)
    np.save(processed_dir / "y.npy",       y)
    np.save(processed_dir / "classes.npy", le.classes_)
    joblib.dump(le, processed_dir / "label_encoder.pkl")

    print(f"Saved features to {processed_dir}/")
    print(f"  X_mfcc : {X_mfcc.shape}")
    print(f"  X_mel  : {X_mel.shape}")
    print(f"  y      : {y.shape}  ({len(le.classes_)} classes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features from BirdCLEF 2023.")
    parser.add_argument(
        "--meta", type=str,
        default="data/processed/train_metadata_filtered.csv",
        help="Path to filtered metadata CSV (output of make_dataset.py)",
    )
    parser.add_argument(
        "--audio-root", type=str, default="data/raw",
        help="Root directory containing train_audio/",
    )
    args = parser.parse_args()

    meta_path  = Path(args.meta)
    audio_root = Path(args.audio_root)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Filtered metadata not found at {meta_path}. "
            "Run scripts/make_dataset.py first."
        )

    df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(df)} rows, {df['primary_label'].nunique()} species")

    X_mfcc, X_mel, labels = build_feature_arrays(df, audio_root)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    save_features(X_mfcc, X_mel, y, le, Path("data/processed"))


if __name__ == "__main__":
    main()
