import './ControlPanel.css'

/**
 * ControlPanel Component
 * Provides layer toggle controls and refresh functionality
 */
function ControlPanel({ 
  showPavement, 
  setShowPavement, 
  showHeatmap, 
  setShowHeatmap,
  showSatellite,
  setShowSatellite,
  onRefresh,
  loading 
}) {
  return (
    <div className="control-panel">
      <h3>🎛️ Layer Controls</h3>
      
      <div className="control-group">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={showSatellite}
            onChange={(e) => setShowSatellite(e.target.checked)}
          />
          <span className="toggle-slider"></span>
          <span className="toggle-text">
            🛰️ Satellite Imagery
          </span>
        </label>
        
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={showPavement}
            onChange={(e) => setShowPavement(e.target.checked)}
          />
          <span className="toggle-slider"></span>
          <span className="toggle-text">
            🛣️ Pavement Markings
          </span>
        </label>
        
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={showHeatmap}
            onChange={(e) => setShowHeatmap(e.target.checked)}
          />
          <span className="toggle-slider"></span>
          <span className="toggle-text">
            🔥 Traffic Heatmap
          </span>
        </label>
      </div>
      
      <div className="control-actions">
        <button 
          className="refresh-btn"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? '⏳ Loading...' : '🔄 Refresh Data'}
        </button>
      </div>
    </div>
  )
}

export default ControlPanel
