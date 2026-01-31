import { useEffect, useRef, useMemo } from 'react'
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons in webpack/vite builds
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

/**
 * HeatmapLayer Component
 * Renders heatmap data as a canvas overlay
 */
function HeatmapLayer({ data, visible }) {
  const map = useMap()
  const canvasLayerRef = useRef(null)
  
  useEffect(() => {
    if (!data || !data.points || data.points.length === 0) return
    if (!visible) {
      if (canvasLayerRef.current) {
        map.removeLayer(canvasLayerRef.current)
        canvasLayerRef.current = null
      }
      return
    }
    
    // Create a canvas layer for the heatmap
    const CanvasHeatmapLayer = L.Layer.extend({
      onAdd: function(map) {
        this._map = map
        this._canvas = L.DomUtil.create('canvas', 'leaflet-heatmap-layer')
        
        const size = map.getSize()
        this._canvas.width = size.x
        this._canvas.height = size.y
        this._canvas.style.position = 'absolute'
        this._canvas.style.top = '0'
        this._canvas.style.left = '0'
        this._canvas.style.pointerEvents = 'none'
        
        map.getPanes().overlayPane.appendChild(this._canvas)
        
        map.on('moveend', this._reset, this)
        map.on('zoomend', this._reset, this)
        map.on('resize', this._resize, this)
        
        this._reset()
      },
      
      onRemove: function(map) {
        L.DomUtil.remove(this._canvas)
        map.off('moveend', this._reset, this)
        map.off('zoomend', this._reset, this)
        map.off('resize', this._resize, this)
      },
      
      _resize: function(e) {
        this._canvas.width = e.newSize.x
        this._canvas.height = e.newSize.y
        this._reset()
      },
      
      _reset: function() {
        const topLeft = this._map.containerPointToLayerPoint([0, 0])
        L.DomUtil.setPosition(this._canvas, topLeft)
        this._draw()
      },
      
      _draw: function() {
        const ctx = this._canvas.getContext('2d')
        const size = this._map.getSize()
        
        ctx.clearRect(0, 0, size.x, size.y)
        
        if (!data.points) return
        
        // Draw heatmap points
        data.points.forEach(point => {
          const latlng = L.latLng(point.lat, point.lon)
          const pixel = this._map.latLngToContainerPoint(latlng)
          
          // Skip points outside viewport
          if (pixel.x < -50 || pixel.x > size.x + 50 ||
              pixel.y < -50 || pixel.y > size.y + 50) {
            return
          }
          
          // Calculate radius based on zoom level
          const zoom = this._map.getZoom()
          const baseRadius = Math.max(8, Math.min(30, zoom * 2))
          const radius = baseRadius * (0.5 + point.intensity * 0.5)
          
          // Create radial gradient
          const gradient = ctx.createRadialGradient(
            pixel.x, pixel.y, 0,
            pixel.x, pixel.y, radius
          )
          
          // Color based on intensity
          const alpha = Math.min(0.8, point.intensity * 0.9)
          const hue = (1 - point.intensity) * 60 // Red (0) to Yellow (60)
          
          gradient.addColorStop(0, `hsla(${hue}, 100%, 50%, ${alpha})`)
          gradient.addColorStop(0.5, `hsla(${hue}, 100%, 50%, ${alpha * 0.5})`)
          gradient.addColorStop(1, `hsla(${hue}, 100%, 50%, 0)`)
          
          ctx.beginPath()
          ctx.arc(pixel.x, pixel.y, radius, 0, Math.PI * 2)
          ctx.fillStyle = gradient
          ctx.fill()
        })
      },
      
      setData: function(newData) {
        data = newData
        this._reset()
      }
    })
    
    // Remove existing layer
    if (canvasLayerRef.current) {
      map.removeLayer(canvasLayerRef.current)
    }
    
    // Add new layer
    const layer = new CanvasHeatmapLayer()
    layer.addTo(map)
    canvasLayerRef.current = layer
    
    return () => {
      if (canvasLayerRef.current) {
        map.removeLayer(canvasLayerRef.current)
        canvasLayerRef.current = null
      }
    }
  }, [map, data, visible])
  
  return null
}

/**
 * MapUpdater Component
 * Updates map view when center/zoom changes
 */
function MapUpdater({ center, zoom }) {
  const map = useMap()
  
  useEffect(() => {
    if (center && center[0] && center[1]) {
      map.setView(center, zoom)
    }
  }, [map, center, zoom])
  
  return null
}

/**
 * Style function for GeoJSON features
 */
function getFeatureStyle(feature) {
  const markingType = feature.properties?.marking_type || 'unknown'
  const confidence = feature.properties?.confidence || 0.5
  
  const colors = {
    lane_line: '#FFFF00',      // Bright Yellow
    crosswalk: '#FF00FF',       // Magenta (visible on satellite)
    stop_line: '#FF0000',       // Red
    arrow: '#00FF00',           // Green
    unknown: '#FFA500'          // Orange
  }
  
  const color = colors[markingType] || colors.unknown
  
  return {
    fillColor: color,
    fillOpacity: 0.7,
    color: color,          // Same color for border
    weight: 3,             // Thicker border for visibility
    opacity: 1.0
  }
}

/**
 * Feature popup content
 */
function onEachFeature(feature, layer) {
  if (feature.properties) {
    const props = feature.properties
    const content = `
      <div class="feature-popup">
        <h4>${props.marking_type?.replace('_', ' ').toUpperCase() || 'Unknown'}</h4>
        <p><strong>Confidence:</strong> ${(props.confidence * 100).toFixed(1)}%</p>
        ${props.area_pixels ? `<p><strong>Area:</strong> ${props.area_pixels.toFixed(0)} px²</p>` : ''}
      </div>
    `
    layer.bindPopup(content)
  }
}

/**
 * Main MapView Component
 */
function MapView({ 
  center, 
  zoom, 
  geojsonData, 
  heatmapData, 
  showPavement, 
  showHeatmap,
  showSatellite 
}) {
  // Memoize GeoJSON key to force re-render when data changes
  const geojsonKey = useMemo(() => {
    return geojsonData ? JSON.stringify(geojsonData).slice(0, 100) : 'empty'
  }, [geojsonData])
  
  return (
    <MapContainer
      center={center}
      zoom={zoom}
      style={{ height: '100%', width: '100%' }}
      zoomControl={true}
      minZoom={16}
      maxZoom={21}
    >
      {/* Base Map Layers */}
      {showSatellite ? (
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
          maxZoom={19}
        />
      ) : (
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          maxZoom={19}
        />
      )}
      
      {/* Labels overlay for satellite view */}
      {showSatellite && (
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
          opacity={0.6}
        />
      )}
      
      {/* Heatmap Layer */}
      <HeatmapLayer 
        data={heatmapData} 
        visible={showHeatmap} 
      />
      
      {/* Pavement Markings GeoJSON Layer */}
      {showPavement && geojsonData && geojsonData.features && (
        <GeoJSON
          key={geojsonKey}
          data={geojsonData}
          style={getFeatureStyle}
          onEachFeature={onEachFeature}
        />
      )}
      
      {/* Map Updater */}
      <MapUpdater center={center} zoom={zoom} />
    </MapContainer>
  )
}

export default MapView
