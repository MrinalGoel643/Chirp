import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Header       from './components/Header'
import AudioUploader from './components/AudioUploader'
import Spectrogram  from './components/Spectrogram'
import BirdResult   from './components/BirdResult'
import RangeMap     from './components/RangeMap'
import { mockPredict, MOCK_BIRDS, SAMPLE_AUDIOS } from './data/mockData'
import { useSpectrogram } from './hooks/useSpectrogram'
import styles from './App.module.css'

export default function App() {
  const [audioSelection, setAudioSelection] = useState(null)  // { file, sampleId }
  const [isLoading,      setIsLoading]      = useState(false)
  const [result,         setResult]         = useState(null)

  const { spectrogramUrl, generateSpectrogram, isGenerating } = useSpectrogram()

  const handleAudioSelected = useCallback((selection) => {
    setAudioSelection(selection)
    setResult(null)
    // Generate spectrogram if a real file was uploaded
    if (selection.file) {
      generateSpectrogram(selection.file)
    }
  }, [generateSpectrogram])

  const handleIdentify = useCallback(async () => {
    if (!audioSelection) return
    setIsLoading(true)

    try {
      // Determine which mock bird to return
      const sampleId = audioSelection.sampleId
        ?? SAMPLE_AUDIOS[Math.floor(Math.random() * SAMPLE_AUDIOS.length)].id

      // ── Replace mockPredict() with real API call when backend is ready ──
      // const res = await fetch('http://localhost:8000/predict', {
      //   method: 'POST',
      //   body: formData,
      // })
      // const data = await res.json()
      const data = await mockPredict(sampleId)
      setResult(data)
    } catch (err) {
      console.error('Prediction failed:', err)
    } finally {
      setIsLoading(false)
    }
  }, [audioSelection])

  const hasAudio = audioSelection && (audioSelection.file || audioSelection.sampleId)

  return (
    <div className={styles.page}>
      {/* Decorative background blobs */}
      <div className={styles.blob1} />
      <div className={styles.blob2} />

      <div className={styles.container}>
        <Header />

        <main className={styles.main}>
          {/* Left panel — always visible */}
          <motion.div
            className={styles.leftPanel}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className={styles.panelCard}>
              <p className={styles.panelTitle}>Listen & Identify</p>
              <p className={styles.panelSub}>
                Upload a bird call recording or try one of our curated samples.
                Chirp will identify the species in seconds.
              </p>

              <div className={styles.divider} />

              <AudioUploader
                onAudioSelected={handleAudioSelected}
                isLoading={isLoading}
              />

              {/* Wire up identify button here */}
              {hasAudio && !isLoading && !result && (
                <motion.button
                  className={styles.identifyBtn}
                  onClick={handleIdentify}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Identify Bird
                </motion.button>
              )}

              {isLoading && (
                <div className={styles.loadingState}>
                  <div className={styles.spinner} />
                  <p>Analysing audio…</p>
                </div>
              )}
            </div>

            {/* Spectrogram below uploader */}
            <AnimatePresence>
              {(spectrogramUrl || isGenerating) && (
                <motion.div
                  className={styles.panelCard}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <Spectrogram url={spectrogramUrl} isGenerating={isGenerating} />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Right panel — results */}
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="placeholder"
                className={`${styles.rightPanel} ${styles.placeholder}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className={styles.placeholderInner}>
                  <div className={styles.birdIllustration}>
                    <svg viewBox="0 0 120 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                      {/* Simple elegant bird silhouette */}
                      <ellipse cx="52" cy="42" rx="22" ry="14" fill="var(--mauve-200)" />
                      <circle cx="74" cy="34" r="10" fill="var(--mauve-200)" />
                      <ellipse cx="30" cy="46" rx="18" ry="8" fill="var(--mauve-100)" transform="rotate(-15 30 46)" />
                      <ellipse cx="68" cy="56" rx="14" ry="5" fill="var(--mauve-100)" transform="rotate(10 68 56)" />
                      {/* Eye */}
                      <circle cx="78" cy="31" r="2.5" fill="var(--ink)" />
                      <circle cx="79" cy="30" r="0.8" fill="white" />
                      {/* Beak */}
                      <path d="M83 34 L92 36 L83 37 Z" fill="var(--gold)" />
                      {/* Branch */}
                      <path d="M10 65 Q60 60 110 68" stroke="var(--beige-300)" strokeWidth="2.5" strokeLinecap="round" fill="none" />
                    </svg>
                  </div>
                  <p className={styles.placeholderText}>
                    Upload or select an audio clip<br />to identify the species
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                className={styles.rightPanel}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              >
                <BirdResult result={result} />
                <RangeMap
                  range={result.range}
                  commonName={result.commonName}
                  color={result.color}
                />

                <button
                  className={styles.resetBtn}
                  onClick={() => { setResult(null); setAudioSelection(null) }}
                >
                  ← Identify another bird
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        <footer className={styles.footer}>
          <p>chirp! · built with BirdCLEF 2023 · EfficientNet-B0</p>
        </footer>
      </div>
    </div>
  )
}
