import React from 'react'
import { motion } from 'framer-motion'
import styles from './Header.module.css'

export default function Header() {
  return (
    <motion.header
      className={styles.header}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className={styles.logo}>
        <span className={styles.logoIcon}>𝄞</span>
        <span className={styles.logoText}>chirp!</span>
      </div>
      <p className={styles.tagline}>your birdwatching companion</p>
      <div className={styles.divider} />
    </motion.header>
  )
}
