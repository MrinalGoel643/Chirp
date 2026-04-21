import React from 'react'
import { motion } from 'framer-motion'
import styles from './Spectrogram.module.css'

export default function Spectrogram({ url, isGenerating }) {
  if (!url && !isGenerating) return null

  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <p className={styles.label}>Audio Spectrogram</p>
      <div className={styles.canvas}>
        {isGenerating ? (
          <div className={styles.generating}>
            <div className={styles.waveform}>
              {Array.from({ length: 24 }).map((_, i) => (
                <div
                  key={i}
                  className={styles.bar}
                  style={{ animationDelay: `${i * 0.05}s` }}
                />
              ))}
            </div>
            <p className={styles.genLabel}>Analysing audio…</p>
          </div>
        ) : (
          <img src={url} alt="Mel spectrogram" className={styles.image} />
        )}
      </div>
      {url && (
        <p className={styles.caption}>
          Mel-frequency spectrogram · time → · frequency ↑
        </p>
      )}
    </motion.div>
  )
}
