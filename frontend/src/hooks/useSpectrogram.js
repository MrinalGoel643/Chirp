import { useState, useCallback } from 'react'

/**
 * Hook to generate a spectrogram canvas from an audio File or Blob.
 * Uses the Web Audio API — no external dependencies.
 *
 * @returns {{ spectrogramUrl, generateSpectrogram, isGenerating }}
 */
export function useSpectrogram() {
  const [spectrogramUrl, setSpectrogramUrl] = useState(null)
  const [isGenerating, setIsGenerating]     = useState(false)

  const generateSpectrogram = useCallback(async (audioFile) => {
    if (!audioFile) return
    setIsGenerating(true)
    setSpectrogramUrl(null)

    try {
      const arrayBuffer = await audioFile.arrayBuffer()
      const audioCtx    = new (window.AudioContext || window.webkitAudioContext)()
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)

      const channelData = audioBuffer.getChannelData(0)
      const sampleRate  = audioBuffer.sampleRate

      // FFT parameters
      const fftSize     = 1024
      const hopSize     = 256
      const numFrames   = Math.floor((channelData.length - fftSize) / hopSize)
      const numFreqBins = fftSize / 2

      // Canvas dimensions
      const width  = Math.min(numFrames, 900)
      const height = 220

      const canvas  = document.createElement('canvas')
      canvas.width  = width
      canvas.height = height
      const ctx     = canvas.getContext('2d')

      // Draw background
      ctx.fillStyle = '#2a1f2a'
      ctx.fillRect(0, 0, width, height)

      // Hanning window
      const window_ = new Float32Array(fftSize)
      for (let i = 0; i < fftSize; i++) {
        window_[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)))
      }

      // Compute magnitude spectrum for each frame
      const frameStep = Math.floor(numFrames / width)

      for (let x = 0; x < width; x++) {
        const frameIdx  = x * frameStep
        const sampleIdx = frameIdx * hopSize

        // Extract windowed frame
        const frame = new Float32Array(fftSize)
        for (let i = 0; i < fftSize; i++) {
          const s = sampleIdx + i < channelData.length ? channelData[sampleIdx + i] : 0
          frame[i] = s * window_[i]
        }

        // Simple DFT magnitude (fast enough for visualization)
        const magnitudes = new Float32Array(numFreqBins)
        for (let k = 0; k < numFreqBins; k++) {
          let re = 0, im = 0
          for (let n = 0; n < fftSize; n++) {
            const angle = (2 * Math.PI * k * n) / fftSize
            re += frame[n] * Math.cos(angle)
            im -= frame[n] * Math.sin(angle)
          }
          magnitudes[k] = Math.sqrt(re * re + im * im)
        }

        // Draw column — map freq bins to height, mel-scale approximation
        for (let y = 0; y < height; y++) {
          const freqFrac = (height - y) / height
          // Mel-like: more resolution at low frequencies
          const binIdx  = Math.floor(Math.pow(freqFrac, 1.8) * numFreqBins)
          const mag     = magnitudes[Math.min(binIdx, numFreqBins - 1)]
          const db      = 20 * Math.log10(mag + 1e-6)
          const norm    = Math.max(0, Math.min(1, (db + 80) / 80))

          // Mauve → gold color ramp
          const r = Math.floor(lerp(42,  210, norm))
          const g = Math.floor(lerp(31,  168, norm))
          const b = Math.floor(lerp(42,   76, norm))

          ctx.fillStyle = `rgb(${r},${g},${b})`
          ctx.fillRect(x, y, 1, 1)
        }
      }

      await audioCtx.close()
      setSpectrogramUrl(canvas.toDataURL('image/png'))
    } catch (err) {
      console.error('Spectrogram generation failed:', err)
    } finally {
      setIsGenerating(false)
    }
  }, [])

  return { spectrogramUrl, generateSpectrogram, isGenerating }
}

function lerp(a, b, t) {
  return a + (b - a) * t
}
