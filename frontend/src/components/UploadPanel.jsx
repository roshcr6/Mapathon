import { useState, useRef, useEffect } from 'react'
import { 
  extractPavement, 
  generateHeatmap, 
  fetchGeoJSON, 
  fetchHeatmap,
  getSatelliteLocations,
  downloadSatellite,
  runCompletePipeline,
  processFootage
} from '../services/api'
import './UploadPanel.css'

/**
 * UploadPanel Component
 * Handles satellite download, pavement extraction, and traffic analysis
 */
function UploadPanel({ onDataUpdate, setLoading, setError, onLocationChange }) {
  const [activeTab, setActiveTab] = useState('pipeline')
  const [locations, setLocations] = useState([
    { id: 'times_square_nyc', name: 'Times Square, NYC' },
    { id: 'shibuya_tokyo', name: 'Shibuya Crossing, Tokyo' },
    { id: 'piccadilly_london', name: 'Piccadilly Circus, London' },
    { id: 'champs_elysees_paris', name: 'Champs-Élysées, Paris' }
  ])
  const [selectedLocation, setSelectedLocation] = useState('times_square_nyc')
  const [imageFile, setImageFile] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  
  const [extractionParams, setExtractionParams] = useState({
    threshold: 220,  // Higher = more selective (only bright lines)
    min_area: 200    // Larger = fewer false positives
  })
  const [heatmapParams, setHeatmapParams] = useState({
    grid_size: 50,
    frame_sample_rate: 2,
    max_frames: 500
  })
  const [processingStatus, setProcessingStatus] = useState('')
  const [pipelineProgress, setPipelineProgress] = useState(0)
  
  const imageInputRef = useRef(null)
  const videoInputRef = useRef(null)
  
  // Fetch available locations on mount
  useEffect(() => {
    async function loadLocations() {
      try {
        const result = await getSatelliteLocations()
        if (result.locations) {
          setLocations(result.locations)
        }
      } catch (err) {
        console.log('Could not load locations, using defaults')
        setLocations([
          { id: 'times_square_nyc', name: 'Times Square, NYC' },
          { id: 'shibuya_tokyo', name: 'Shibuya Crossing, Tokyo' },
          { id: 'piccadilly_london', name: 'Piccadilly Circus, London' },
          { id: 'champs_elysees_paris', name: 'Champs-Élysées, Paris' }
        ])
      }
    }
    loadLocations()
  }, [])
  
  // Auto-run pipeline on component mount
  useEffect(() => {
    // Auto-run pipeline after locations are loaded
    const timer = setTimeout(() => {
      handleRunPipeline()
    }, 1000)
    
    return () => clearTimeout(timer)
  }, []) // Run once on mount
  
  // Run complete AI pipeline
  const handleRunPipeline = async () => {
    setLoading(true)
    setError(null)
    setPipelineProgress(0)
    
    try {
      // Step 1: Download satellite
      setProcessingStatus('📡 Downloading satellite imagery...')
      setPipelineProgress(10)
      
      // Step 2-4: Run complete pipeline
      setProcessingStatus('🤖 Running AI analysis pipeline on real data...')
      setPipelineProgress(30)
      
      const result = await runCompletePipeline({
        location: selectedLocation,
        processVideo: true,
        threshold: extractionParams.threshold,
        gridSize: heatmapParams.grid_size,
        maxFrames: heatmapParams.max_frames
      })
      
      setPipelineProgress(80)
      
      // Check if we got valid results
      if (result && result.success !== false) {
        setProcessingStatus('🗺️ Loading results...')
        
        // Fetch results
        const [geojson, heatmap] = await Promise.all([
          fetchGeoJSON(),
          fetchHeatmap()
        ])
        
        onDataUpdate('geojson', geojson)
        onDataUpdate('heatmap', heatmap)
        
        // Update map location with higher zoom for detail
        if (result.bounds && onLocationChange) {
          const center = {
            lat: (result.bounds.min_lat + result.bounds.max_lat) / 2,
            lng: (result.bounds.min_lon + result.bounds.max_lon) / 2
          }
          onLocationChange(center, 18)  // Zoom level 18 for better detail
        }
        
        setPipelineProgress(100)
        setProcessingStatus(`✅ Complete! Extracted ${result.pavement_features || 0} lane markings, processed ${result.frames_processed || 0} video frames from real CCTV`)
      } else {
        throw new Error(result.message || 'Pipeline failed')
      }
    } catch (err) {
      setError(`Pipeline failed: ${err.message}`)
      setProcessingStatus('')
    } finally {
      setLoading(false)
    }
  }
  
  // Process local footage only
  const handleProcessFootage = async () => {
    setLoading(true)
    setError(null)
    setProcessingStatus('🎥 Processing local CCTV footage...')
    
    try {
      const result = await processFootage({
        gridSize: heatmapParams.grid_size,
        frameSampleRate: heatmapParams.frame_sample_rate,
        maxFrames: heatmapParams.max_frames
      })
      
      if (result.status === 'success') {
        const heatmap = await fetchHeatmap()
        onDataUpdate('heatmap', heatmap)
        setProcessingStatus(`✅ Processed ${result.frames_processed} frames, detected ${result.vehicles_detected} vehicles`)
      } else {
        throw new Error(result.message || 'Processing failed')
      }
    } catch (err) {
      setError(`Video processing failed: ${err.message}`)
      setProcessingStatus('')
    } finally {
      setLoading(false)
    }
  }
  
  // Download satellite imagery
  const handleDownloadSatellite = async () => {
    setLoading(true)
    setError(null)
    setProcessingStatus('📡 Downloading satellite tiles...')
    
    try {
      const result = await downloadSatellite(selectedLocation)
      
      if (result.status === 'success') {
        setProcessingStatus(`✅ Downloaded ${result.tiles_downloaded} tiles for ${result.location}`)
        
        // Update map location
        if (result.bounds && onLocationChange) {
          const center = {
            lat: (result.bounds.min_lat + result.bounds.max_lat) / 2,
            lng: (result.bounds.min_lon + result.bounds.max_lon) / 2
          }
          onLocationChange(center, 17)
        }
      }
    } catch (err) {
      setError(`Satellite download failed: ${err.message}`)
      setProcessingStatus('')
    } finally {
      setLoading(false)
    }
  }
  
  // Manual image upload
  const handleImageUpload = async () => {
    if (!imageFile) {
      setError('Please select a satellite image file')
      return
    }
    
    setLoading(true)
    setProcessingStatus('🔍 Extracting pavement markings with AI...')
    setError(null)
    
    try {
      const result = await extractPavement(imageFile, extractionParams)
      
      if (result.success) {
        setProcessingStatus(`✅ Extracted ${result.feature_count} features in ${result.processing_time?.toFixed(2)}s`)
        const geojson = await fetchGeoJSON()
        onDataUpdate('geojson', geojson)
      }
    } catch (err) {
      setError(`Extraction failed: ${err.message}`)
      setProcessingStatus('')
    } finally {
      setLoading(false)
    }
  }
  
  // Manual video upload
  const handleVideoUpload = async () => {
    if (!videoFile) {
      setError('Please select a CCTV video file')
      return
    }
    
    setLoading(true)
    setProcessingStatus('🔥 Generating traffic heatmap...')
    setError(null)
    
    try {
      const result = await generateHeatmap(videoFile, heatmapParams)
      
      if (result.success) {
        setProcessingStatus(`✅ Processed ${result.total_frames_processed} frames in ${result.processing_time?.toFixed(2)}s`)
        const heatmap = await fetchHeatmap()
        onDataUpdate('heatmap', heatmap)
      }
    } catch (err) {
      setError(`Heatmap generation failed: ${err.message}`)
      setProcessingStatus('')
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="upload-panel">
      <h3>🚀 AI Traffic Analysis System</h3>
      <p className="system-description">Automated pavement detection and traffic heatmap from real data</p>
      
      {/* Pipeline Status Display */}
      <div className="tab-content">
        <div className="upload-section">
          <h4>📊 Processing Status</h4>
          
          {processingStatus && (
            <div className="processing-status">
              <p>{processingStatus}</p>
              {pipelineProgress > 0 && (
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${pipelineProgress}%` }}
                  />
                </div>
              )}
            </div>
          )}
          
          <div className="param-group">
            <label htmlFor="threshold-slider">
              Detection Threshold (higher = only bright lines):
              <input
                id="threshold-slider"
                name="threshold"
                type="range"
                min="180"
                max="245"
                value={extractionParams.threshold}
                onChange={(e) => setExtractionParams(prev => ({
                  ...prev,
                  threshold: parseInt(e.target.value)
                }))}
              />
              <span className="param-value">{extractionParams.threshold}</span>
            </label>
          </div>
          
          <div className="param-group">
            <label htmlFor="maxframes-slider">
              Max Video Frames to Process:
              <input
                id="maxframes-slider"
                name="max_frames"
                type="range"
                min="100"
                max="1000"
                step="50"
                value={heatmapParams.max_frames}
                onChange={(e) => setHeatmapParams(prev => ({
                  ...prev,
                  max_frames: parseInt(e.target.value)
                }))}
              />
              <span className="param-value">{heatmapParams.max_frames}</span>
            </label>
          </div>
          
          <button 
            className="primary-btn"
            onClick={handleRunPipeline}
          >
            🔄 Re-run AI Pipeline
          </button>
          
          <div className="info-box">
            <h4>ℹ️ System Info</h4>
            <p>• Using REAL satellite imagery from ESRI</p>
            <p>• Processing REAL CCTV footage: vecteezy_image-of-traffic...mov</p>
            <p>• AI trained to detect lane lines (even faded)</p>
            <p>• No demo/fake data used</p>
          </div>
        </div>
      </div>
      
      {processingStatus && (
        <div className="status-message">
          {processingStatus}
        </div>
      )}
    </div>
  )
}

export default UploadPanel
