"""
scripts/make_dataset.py

Downloads BirdCLEF 2023 from Kaggle and filters train_metadata.csv
to the top N North American species by sample count.

Usage:
    python scripts/make_dataset.py --top-n 20 --min-samples 30

Attribution:
    BirdCLEF 2023 dataset — Cornell Lab of Ornithology
    https://www.kaggle.com/competitions/birdclef-2023
"""

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


# North America approximate bounding box
NA_LAT_MIN, NA_LAT_MAX =  15.0,  72.0
NA_LON_MIN, NA_LON_MAX = -168.0, -52.0


def download_dataset(data_dir: Path) -> None:
    """
    Download BirdCLEF 2023 via the Kaggle CLI into data/raw/.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables
    or a valid ~/.kaggle/kaggle.json credential file.

    Args:
        data_dir: Destination directory for the downloaded zip.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading BirdCLEF 2023 from Kaggle...")
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "birdclef-2023",
            "-p", str(data_dir),
        ],
        check=True,
    )

    zip_path = data_dir / "birdclef-2023.zip"
    if zip_path.exists():
        print("Extracting archive...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(data_dir)], check=True)
        zip_path.unlink()
        print(f"Extracted to {data_dir}")
    else:
        print("Zip not found — data may already be extracted.")


def filter_north_america(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter recordings to those geotagged within North America.

    Uses a lat/lon bounding box: lat [15, 72], lon [-168, -52].
    Rows without coordinates are dropped.

    Args:
        df: Raw train_metadata DataFrame.

    Returns:
        Filtered DataFrame.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("Warning: no lat/lon columns found — skipping geographic filter.")
        return df

    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        df["latitude"].between(NA_LAT_MIN, NA_LAT_MAX) &
        df["longitude"].between(NA_LON_MIN, NA_LON_MAX)
    ]
    print(f"Geographic filter: {before} → {len(df)} rows (North America only)")
    return df.reset_index(drop=True)


def select_top_species(df: pd.DataFrame, top_n: int, min_samples: int) -> pd.DataFrame:
    """
    Keep only the top N species by sample count, each with at least
    min_samples recordings.

    Args:
        df:          Filtered metadata DataFrame.
        top_n:       Number of species to retain.
        min_samples: Minimum recordings required per species.

    Returns:
        Further-filtered DataFrame.
    """
    counts = df["primary_label"].value_counts()
    valid  = counts[counts >= min_samples].head(top_n).index.tolist()
    df     = df[df["primary_label"].isin(valid)].reset_index(drop=True)

    print(f"Selected {df['primary_label'].nunique()} species, {len(df)} total recordings.")
    print("Species included:")
    for sp, cnt in df["primary_label"].value_counts().items():
        print(f"  {sp}: {cnt}")

    return df


def save_filtered_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the filtered metadata CSV to the processed data directory.

    Args:
        df:          Filtered metadata DataFrame.
        output_path: Destination .csv path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved filtered metadata → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and scope BirdCLEF 2023 dataset.")
    parser.add_argument("--top-n",      type=int, default=20, help="Number of top species to keep")
    parser.add_argument("--min-samples",type=int, default=30, help="Minimum samples per species")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download if data already exists")
    args = parser.parse_args()

    raw_dir       = Path("data/raw")
    processed_dir = Path("data/processed")
    meta_path     = raw_dir / "train_metadata.csv"
    out_path      = processed_dir / "train_metadata_filtered.csv"

    if not args.skip_download:
        download_dataset(raw_dir)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. "
            "Run without --skip-download or check your data/raw/ directory."
        )

    df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(df)} rows, {df['primary_label'].nunique()} species")

    df = filter_north_america(df)
    df = select_top_species(df, top_n=args.top_n, min_samples=args.min_samples)
    save_filtered_metadata(df, out_path)


if __name__ == "__main__":
    main()
