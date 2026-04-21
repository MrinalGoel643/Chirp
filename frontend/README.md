# chirp! 🐦
### your birdwatching companion

A minimalist bird species identification app powered by audio analysis. Upload a bird call recording and Chirp identifies the species using a fine-tuned EfficientNet-B0 model trained on BirdCLEF 2023.

## Features
- Upload your own bird audio or try curated samples
- Real-time mel spectrogram visualization (Web Audio API)
- Species identification with confidence scores
- Bird facts, habitat, diet
- North American range map (Leaflet)

## Stack
- **Frontend**: Vite + React + Framer Motion + Leaflet
- **Backend**: FastAPI + PyTorch (EfficientNet-B0)
- **Model**: BirdCLEF 2023, top 20 North American species

## Getting Started

```bash
npm install
npm run dev
```

## Connecting the Backend

In `src/App.jsx`, replace `mockPredict()` with a real fetch:

```js
const formData = new FormData()
formData.append('file', audioSelection.file)

const res = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
})
const data = await res.json()
```

The backend should return JSON matching the shape in `src/data/mockData.js`.
