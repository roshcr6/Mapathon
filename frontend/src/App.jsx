import { useState, useEffect } from 'react'
import MapView from './components/MapView'
import ControlPanel from './components/ControlPanel'
import UploadPanel from './components/UploadPanel'
import Legend from './components/Legend'
import { fetchGeoJSON, fetchHeatmap, generateDemoHeatmap, checkHealth, runCompletePipeline } from './services/api'
import './styles/App.css'

function App() {
  // State for layer visibility
  const [showPavement, setShowPavement] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [showSatellite, setShowSatellite] = useState(true)
  
  // State for data
  const [geojsonData, setGeojsonData] = useState(null)
  const [heatmapData, setHeatmapData] = useState(null)
  
  // State for UI
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mapCenter, setMapCenter] = useState([40.7580, -73.9855]) // Times Square
  const [mapZoom, setMapZoom] = useState(18)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [appReady, setAppReady] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [detectionStats, setDetectionStats] = useState(null)

  // Check backend health and load data on mount
  useEffect(() => {
    // Small delay to ensure DOM is ready
    setTimeout(() => {
      setAppReady(true)
      checkBackendAndLoad()
    }, 100)
  }, [])

  const checkBackendAndLoad = async () => {
    setLoading(true)
    setStatusMessage('🔌 Connecting to AI backend...')
    
    try {
      const health = await checkHealth()
      if (health.healthy) {
        setBackendStatus('connected')
        setStatusMessage('🤖 Running AI detection pipeline...')
        await runAIPipeline()
      } else {
        setBackendStatus('disconnected')
        setStatusMessage('❌ Backend not connected')
      }
    } catch (err) {
      console.log('Backend not connected yet:', err.message)
      setBackendStatus('disconnected')
      setStatusMessage('❌ Cannot connect to backend - start the server first')
    } finally {
      setLoading(false)
    }
  }

  const runAIPipeline = async () => {
    try {
      setStatusMessage('📡 Downloading satellite imagery for Times Square...')
      
      // Run the complete pipeline
      const result = await runCompletePipeline({
        location: 'times_square_nyc',
        processVideo: true,
        threshold: 200,
        gridSize: 50,
        maxFrames: 300
      })
      
      setStatusMessage('🗺️ Loading AI detection results...')
      
      // Fetch the results
      const [geojson, heatmap] = await Promise.all([
        fetchGeoJSON(),
        fetchHeatmap()
      ])
      
      setGeojsonData(geojson)
      setHeatmapData(heatmap)
      
      // Count detections by type
      const typeCounts = {}
      if (geojson?.features) {
        geojson.features.forEach(f => {
          const t = f.properties?.marking_type || 'unknown'
          typeCounts[t] = (typeCounts[t] || 0) + 1
        })
      }
      
      setDetectionStats({
        totalFeatures: geojson?.features?.length || 0,
        typeCounts,
        heatmapPoints: heatmap?.points?.length || 0,
        framesProcessed: result?.frames_processed || 0
      })
      
      // Update map center
      if (result?.bounds) {
        const centerLat = (result.bounds.min_lat + result.bounds.max_lat) / 2
        const centerLon = (result.bounds.min_lon + result.bounds.max_lon) / 2
        setMapCenter([centerLat, centerLon])
        setMapZoom(18)
      }
      
      setStatusMessage(`✅ Detected ${geojson?.features?.length || 0} pavement markings (${typeCounts.lane_line || 0} lane lines, ${typeCounts.crosswalk || 0} crosswalks)`)
      
    } catch (err) {
      console.error('Pipeline error:', err)
      setError('Pipeline failed: ' + err.message)
      setStatusMessage('')
    }
  }

  const handleDataUpdate = async (type, newData) => {
    if (type === 'geojson') {
      setGeojsonData(newData)
      if (newData?.metadata?.geo_bounds) {
        const bounds = newData.metadata.geo_bounds
        const centerLat = (bounds.min_lat + bounds.max_lat) / 2
        const centerLon = (bounds.min_lon + bounds.max_lon) / 2
        setMapCenter([centerLat, centerLon])
      }
    } else if (type === 'heatmap') {
      setHeatmapData(newData)
      if (newData?.bounds) {
        const bounds = newData.bounds
        const centerLat = (bounds.min_lat + bounds.max_lat) / 2
        const centerLon = (bounds.min_lon + bounds.max_lon) / 2
        setMapCenter([centerLat, centerLon])
      }
    }
  }

  const handleLocationChange = (center, zoom) => {
    setMapCenter([center.lat, center.lng])
    if (zoom) setMapZoom(zoom)
  }

  // Show loading screen while app initializes
  if (!appReady) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#0f0f23',
        color: 'white',
        fontSize: '20px'
      }}>
        <div>
          <div className="spinner" style={{margin: '0 auto 20px'}}></div>
          <p>Loading Mapathon...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🗺️ Mapathon</h1>
        <p>AI-Powered Pavement Marking Extraction & Traffic Heatmap Analysis</p>
        <div className={`backend-status ${backendStatus}`}>
          {backendStatus === 'connected' ? '🟢 Backend Connected' : 
           backendStatus === 'checking' ? '🟡 Connecting...' : '🔴 Backend Offline'}
        </div>
      </header>
      
      <main className="app-main">
        <aside className="sidebar">
          <ControlPanel
            showPavement={showPavement}
            setShowPavement={setShowPavement}
            showHeatmap={showHeatmap}
            setShowHeatmap={setShowHeatmap}
            showSatellite={showSatellite}
            setShowSatellite={setShowSatellite}
            onRefresh={runAIPipeline}
            loading={loading}
          />
          
          <UploadPanel
            onDataUpdate={handleDataUpdate}
            setLoading={setLoading}
            setError={setError}
            onLocationChange={handleLocationChange}
          />
          
          <Legend
            geojsonData={geojsonData}
            heatmapData={heatmapData}
          />
          
          {statusMessage && (
            <div style={{
              margin: '10px',
              padding: '10px',
              background: '#1a1a2e',
              border: '1px solid #00ff88',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#00ff88'
            }}>
              {statusMessage}
            </div>
          )}
          
          {detectionStats && (
            <div style={{
              margin: '10px',
              padding: '10px',
              background: '#1a1a2e',
              border: '1px solid #ffd700',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#ffd700'
            }}>
              <div>📊 Detection Results:</div>
              <div>✓ {detectionStats.totalFeatures} total markings</div>
              <div>✓ {detectionStats.typeCounts.lane_line || 0} lane lines</div>
              <div>✓ {detectionStats.typeCounts.crosswalk || 0} crosswalks</div>
              <div>✓ {detectionStats.heatmapPoints} heatmap points</div>
            </div>
          )}
        </aside>
        
        <div className="map-container">
          {error && (
            <div className="error-banner">
              {error}
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}
          
          {loading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p>Processing...</p>
            </div>
          )}
          
          <MapView
            center={mapCenter}
            zoom={mapZoom}
            geojsonData={geojsonData}
            heatmapData={heatmapData}
            showPavement={showPavement}
            showHeatmap={showHeatmap}
            showSatellite={showSatellite}
          />
        </div>
      </main>
      
      <footer className="app-footer">
        <p>Mapathon Demo System - Pavement Extraction & Traffic Analysis</p>
      </footer>
    </div>
  )
}

export default App
