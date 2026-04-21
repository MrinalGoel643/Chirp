# 🐦 Chirp — Bird Audio Species Classifier

Identify North American bird species from short audio recordings.  
Trained on BirdCLEF 2023 · EfficientNet-B0 · 0.68 test accuracy (top 10 species)

---

## AI usage

Claude Sonnet 4.6 was used for brainstorming and code generation. 

## Models

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Naive Baseline | 0.28 | 0.11 |
| Random Forest (MFCC) | 0.55 | 0.50 |
| **EfficientNet-B0** | **0.68** | **0.65** |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```bash
# Set Kaggle credentials first
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

python scripts/make_dataset.py --top-n 20 --min-samples 30
```

### 3. Extract features
```bash
python scripts/build_features.py
```

### 4. Train all models
```bash
python setup.py --epochs 20
```

### 5. Run the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API

### `POST /predict`
Upload an audio file and receive top-3 species predictions.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_bird_recording.ogg"
```

**Response:**
```json
{
  "predictions": [
    { "species_code": "norcar", "confidence": 0.92 },
    { "species_code": "houspa", "confidence": 0.05 },
    { "species_code": "amered", "confidence": 0.03 }
  ],
  "top_species": "norcar",
  "confidence": 0.92,
  "model": "EfficientNet-B0"
}
```

### `GET /health`
### `GET /classes`

---

## HuggingFace Deployment

Upload model artifacts after training:
```python
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="models/", repo_id="your-username/warbler")
```

Load at runtime by setting:
```bash
export HF_REPO_ID=your-username/warbler
```

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── setup.py               ← full training pipeline
├── app.py                 ← FastAPI inference server
├── scripts/
│   ├── make_dataset.py    ← download + scope BirdCLEF 2023
│   ├── build_features.py  ← extract MFCC + mel spectrogram features
│   └── model.py           ← NaiveBaseline, RandomForestModel, EfficientNetModel
├── models/                ← saved weights + label_encoder.pkl + model_config.json
├── data/
│   ├── raw/               ← BirdCLEF 2023 audio + train_metadata.csv
│   ├── processed/         ← X_mfcc.npy, X_mel.npy, y.npy
│   └── outputs/           ← model_comparison.csv, plots
├── notebooks/             ← exploratory training notebook (not graded)
└── .gitignore
```
