import './Legend.css'

/**
 * Legend Component
 * Displays information about the map layers and data statistics
 */
function Legend({ geojsonData, heatmapData }) {
  // Calculate statistics
  const featureCount = geojsonData?.features?.length || 0
  const heatmapPoints = heatmapData?.points?.length || 0
  
  // Count features by type
  const featuresByType = {}
  if (geojsonData?.features) {
    geojsonData.features.forEach(f => {
      const type = f.properties?.marking_type || 'unknown'
      featuresByType[type] = (featuresByType[type] || 0) + 1
    })
  }
  
  const markingTypes = [
    { type: 'lane_line', label: 'Lane Lines', color: '#FFFF00' },
    { type: 'crosswalk', label: 'Crosswalks', color: '#FFFFFF' },
    { type: 'stop_line', label: 'Stop Lines', color: '#FF0000' },
    { type: 'arrow', label: 'Arrows', color: '#00FF00' },
    { type: 'unknown', label: 'Other', color: '#FFA500' }
  ]
  
  return (
    <div className="legend">
      <h3>📊 Legend</h3>
      
      {/* Pavement Markings Legend */}
      <div className="legend-section">
        <h4>Pavement Markings</h4>
        <div className="legend-items">
          {markingTypes.map(({ type, label, color }) => (
            <div key={type} className="legend-item">
              <span 
                className="legend-color" 
                style={{ backgroundColor: color }}
              ></span>
              <span className="legend-label">{label}</span>
              <span className="legend-count">
                {featuresByType[type] || 0}
              </span>
            </div>
          ))}
        </div>
        <div className="legend-total">
          Total Features: <strong>{featureCount}</strong>
        </div>
      </div>
      
      {/* Heatmap Legend */}
      <div className="legend-section">
        <h4>Traffic Heatmap</h4>
        <div className="heatmap-gradient">
          <div className="gradient-bar"></div>
          <div className="gradient-labels">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
        <div className="legend-stats">
          <div className="stat-item">
            <span>Data Points:</span>
            <strong>{heatmapPoints}</strong>
          </div>
          {heatmapData?.statistics && (
            <>
              <div className="stat-item">
                <span>Max Intensity:</span>
                <strong>{(heatmapData.statistics.max_intensity * 100).toFixed(1)}%</strong>
              </div>
              <div className="stat-item">
                <span>Frames Processed:</span>
                <strong>{heatmapData.statistics.frames_processed || 'N/A'}</strong>
              </div>
            </>
          )}
        </div>
      </div>
      
      {/* Data Source Info */}
      <div className="legend-section data-info">
        <h4>📍 Data Source</h4>
        {geojsonData?.metadata?.demo && (
          <p className="demo-badge">🎭 Demo Data</p>
        )}
        {geojsonData?.metadata?.geo_bounds && (
          <div className="bounds-info">
            <p>Lat: {geojsonData.metadata.geo_bounds.min_lat?.toFixed(4)} - {geojsonData.metadata.geo_bounds.max_lat?.toFixed(4)}</p>
            <p>Lon: {geojsonData.metadata.geo_bounds.min_lon?.toFixed(4)} - {geojsonData.metadata.geo_bounds.max_lon?.toFixed(4)}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Legend
