import React from 'react'
import { motion } from 'framer-motion'
import styles from './BirdResult.module.css'

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
}

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
}

export default function BirdResult({ result }) {
  if (!result) return null

  const { commonName, scientificName, confidence, family, habitat, diet, funFacts, topPredictions, wikiImage, color } = result

  return (
    <motion.div
      className={styles.wrapper}
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      {/* Hero — name + image */}
      <motion.div className={styles.hero} variants={itemVariants}>
        <div className={styles.heroText}>
          <p className={styles.detected}>Species detected</p>
          <h2 className={styles.commonName}>{commonName}</h2>
          <p className={styles.sciName}>{scientificName}</p>
          <div className={styles.confidencePill} style={{ background: color + '22', color }}>
            {Math.round(confidence * 100)}% confidence
          </div>
        </div>
        {wikiImage && (
          <div className={styles.heroImage}>
            <img src={wikiImage} alt={commonName} />
          </div>
        )}
      </motion.div>

      {/* Prediction confidence bars */}
      <motion.div className={styles.card} variants={itemVariants}>
        <p className={styles.cardLabel}>Top Predictions</p>
        <div className={styles.bars}>
          {topPredictions.map((pred, i) => (
            <div key={i} className={styles.barRow}>
              <span className={styles.barName}>{pred.name}</span>
              <div className={styles.barTrack}>
                <motion.div
                  className={styles.barFill}
                  style={{ background: i === 0 ? color : 'var(--mauve-200)' }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pred.confidence * 100}%` }}
                  transition={{ duration: 0.7, delay: 0.2 + i * 0.1, ease: 'easeOut' }}
                />
              </div>
              <span className={styles.barPct}>{Math.round(pred.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Quick facts */}
      <motion.div className={styles.card} variants={itemVariants}>
        <p className={styles.cardLabel}>Field Notes</p>
        <div className={styles.factGrid}>
          <div className={styles.factItem}>
            <span className={styles.factKey}>Family</span>
            <span className={styles.factVal}>{family}</span>
          </div>
          <div className={styles.factItem}>
            <span className={styles.factKey}>Habitat</span>
            <span className={styles.factVal}>{habitat}</span>
          </div>
          <div className={styles.factItem}>
            <span className={styles.factKey}>Diet</span>
            <span className={styles.factVal}>{diet}</span>
          </div>
        </div>
      </motion.div>

      {/* Fun facts */}
      <motion.div className={styles.card} variants={itemVariants}>
        <p className={styles.cardLabel}>Did you know?</p>
        <ul className={styles.factsList}>
          {funFacts.map((fact, i) => (
            <li key={i} className={styles.factLine}>
              <span className={styles.factDot} style={{ background: color }} />
              {fact}
            </li>
          ))}
        </ul>
      </motion.div>
    </motion.div>
  )
}
