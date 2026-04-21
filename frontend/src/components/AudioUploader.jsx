import React, { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Music, ChevronRight } from 'lucide-react'
import { SAMPLE_AUDIOS } from '../data/mockData'
import styles from './AudioUploader.module.css'

export default function AudioUploader({ onAudioSelected, isLoading }) {
  const [activeTab, setActiveTab]   = useState('upload') // 'upload' | 'samples'
  const [dragOver, setDragOver]     = useState(false)
  const [fileName, setFileName]     = useState(null)
  const [activeSample, setActiveSample] = useState(null)
  const fileRef = useRef()

  function handleFile(file) {
    if (!file) return
    setFileName(file.name)
    setActiveSample(null)
    onAudioSelected({ file, sampleId: null })
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('audio')) handleFile(file)
  }

  function handleSample(sample) {
    setActiveSample(sample.id)
    setFileName(null)
    onAudioSelected({ file: null, sampleId: sample.id })
  }

  return (
    <div className={styles.wrapper}>
      {/* Tab switcher */}
      <div className={styles.tabs}>
        {['upload', 'samples'].map(tab => (
          <button
            key={tab}
            className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'upload' ? 'Upload Audio' : 'Try a Sample'}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'upload' ? (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            {/* Drop zone */}
            <div
              className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''} ${fileName ? styles.hasFile : ''}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept="audio/*"
                style={{ display: 'none' }}
                onChange={e => handleFile(e.target.files[0])}
              />
              <div className={styles.dropContent}>
                {fileName ? (
                  <>
                    <Music size={28} color="var(--mauve-400)" />
                    <p className={styles.fileName}>{fileName}</p>
                    <p className={styles.dropHint}>Click to change file</p>
                  </>
                ) : (
                  <>
                    <Upload size={28} color="var(--mauve-300)" />
                    <p className={styles.dropLabel}>Drop your bird audio here</p>
                    <p className={styles.dropHint}>.mp3, .ogg, .wav, .flac supported</p>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="samples"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className={styles.sampleList}
          >
            {SAMPLE_AUDIOS.map(sample => (
              <button
                key={sample.id}
                className={`${styles.sampleCard} ${activeSample === sample.id ? styles.sampleActive : ''}`}
                onClick={() => handleSample(sample)}
              >
                <span className={styles.sampleEmoji}>{sample.emoji}</span>
                <div className={styles.sampleInfo}>
                  <span className={styles.sampleName}>{sample.label}</span>
                  <span className={styles.sampleDesc}>{sample.description}</span>
                </div>
                <ChevronRight size={16} color="var(--mauve-300)" />
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  )
}
