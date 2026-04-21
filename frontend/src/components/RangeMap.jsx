import React, { useEffect } from 'react'
import { motion } from 'framer-motion'
import styles from './RangeMap.module.css'

// Lazy-load Leaflet to avoid SSR issues
let L, MapContainer, TileLayer, Circle, Popup

async function loadLeaflet() {
  if (typeof window === 'undefined') return false
  L = await import('leaflet')
  const rl = await import('react-leaflet')
  MapContainer = rl.MapContainer
  TileLayer    = rl.TileLayer
  Circle       = rl.Circle
  Popup        = rl.Popup
  return true
}

export default function RangeMap({ range, commonName, color }) {
  const [ready, setReady] = React.useState(false)

  useEffect(() => {
    loadLeaflet().then(setReady)
  }, [])

  if (!range) return null

  const { center, description } = range

  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <p className={styles.label}>Range Map</p>
      <div className={styles.mapContainer}>
        {ready && MapContainer ? (
          <MapContainer
            center={center}
            zoom={3}
            scrollWheelZoom={false}
            style={{ height: '100%', width: '100%' }}
            zoomControl={false}
          >
            <TileLayer
              attribution=""
              url="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
            />
            <Circle
              center={center}
              radius={800000}
              pathOptions={{
                color: color || '#9a749a',
                fillColor: color || '#9a749a',
                fillOpacity: 0.18,
                weight: 1.5,
              }}
            >
              <Popup>{commonName} — breeding range</Popup>
            </Circle>
          </MapContainer>
        ) : (
          <div className={styles.mapPlaceholder}>Loading map…</div>
        )}
      </div>
      <p className={styles.rangeDesc}>{description}</p>
    </motion.div>
  )
}
