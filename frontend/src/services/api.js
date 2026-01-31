/**
 * API Service for Mapathon Frontend
 * Handles communication with the backend API
 */

const API_BASE_URL = '/api';

/**
 * Fetch GeoJSON data for pavement markings
 */
export async function fetchGeoJSON() {
  try {
    const response = await fetch(`${API_BASE_URL}/get-geojson`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching GeoJSON:', error);
    throw error;
  }
}

/**
 * Fetch heatmap data
 */
export async function fetchHeatmap() {
  try {
    const response = await fetch(`${API_BASE_URL}/get-heatmap`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching heatmap:', error);
    throw error;
  }
}

/**
 * Get available satellite locations
 */
export async function getSatelliteLocations() {
  try {
    const response = await fetch(`${API_BASE_URL}/satellite/locations`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching locations:', error);
    throw error;
  }
}

/**
 * Download satellite imagery for a location
 */
export async function downloadSatellite(location, customBounds = null, zoom = 18) {
  const formData = new FormData();
  formData.append('location', location);
  formData.append('zoom', zoom.toString());
  
  if (customBounds) {
    formData.append('custom_min_lat', customBounds.min_lat.toString());
    formData.append('custom_max_lat', customBounds.max_lat.toString());
    formData.append('custom_min_lon', customBounds.min_lon.toString());
    formData.append('custom_max_lon', customBounds.max_lon.toString());
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/satellite/download`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error downloading satellite:', error);
    throw error;
  }
}

/**
 * Run complete pipeline: satellite + pavement + traffic
 */
export async function runCompletePipeline(options = {}) {
  const formData = new FormData();
  formData.append('location', options.location || 'times_square_nyc');
  formData.append('process_video', (options.processVideo !== false).toString());
  formData.append('threshold', (options.threshold || 200).toString());
  formData.append('grid_size', (options.gridSize || 50).toString());
  formData.append('max_frames', (options.maxFrames || 500).toString());
  
  try {
    const response = await fetch(`${API_BASE_URL}/run-complete-pipeline`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.text();
      let errorMessage = `Server error ${response.status}`;
      
      try {
        const errorJson = JSON.parse(errorData);
        errorMessage = errorJson.detail || errorJson.message || errorMessage;
      } catch (e) {
        errorMessage = errorData || errorMessage;
      }
      
      throw new Error(errorMessage);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error running pipeline:', error);
    throw error;
  }
}

/**
 * Process existing footage in the footage folder
 */
export async function processFootage(options = {}) {
  const formData = new FormData();
  formData.append('grid_size', (options.gridSize || 50).toString());
  formData.append('frame_sample_rate', (options.frameSampleRate || 2).toString());
  formData.append('max_frames', (options.maxFrames || 500).toString());
  
  if (options.bounds) {
    formData.append('min_lat', options.bounds.min_lat.toString());
    formData.append('max_lat', options.bounds.max_lat.toString());
    formData.append('min_lon', options.bounds.min_lon.toString());
    formData.append('max_lon', options.bounds.max_lon.toString());
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/process-footage`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error processing footage:', error);
    throw error;
  }
}

/**
 * Upload satellite image for pavement extraction
 */
export async function extractPavement(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);
  
  if (options.threshold !== undefined) {
    formData.append('threshold', options.threshold.toString());
  }
  if (options.min_area !== undefined) {
    formData.append('min_area', options.min_area.toString());
  }
  if (options.min_lat !== undefined) {
    formData.append('min_lat', options.min_lat.toString());
  }
  if (options.max_lat !== undefined) {
    formData.append('max_lat', options.max_lat.toString());
  }
  if (options.min_lon !== undefined) {
    formData.append('min_lon', options.min_lon.toString());
  }
  if (options.max_lon !== undefined) {
    formData.append('max_lon', options.max_lon.toString());
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/extract-pavement`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error extracting pavement:', error);
    throw error;
  }
}

/**
 * Upload video for heatmap generation
 */
export async function generateHeatmap(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);
  
  if (options.grid_size !== undefined) {
    formData.append('grid_size', options.grid_size.toString());
  }
  if (options.frame_sample_rate !== undefined) {
    formData.append('frame_sample_rate', options.frame_sample_rate.toString());
  }
  if (options.min_lat !== undefined) {
    formData.append('min_lat', options.min_lat.toString());
  }
  if (options.max_lat !== undefined) {
    formData.append('max_lat', options.max_lat.toString());
  }
  if (options.min_lon !== undefined) {
    formData.append('min_lon', options.min_lon.toString());
  }
  if (options.max_lon !== undefined) {
    formData.append('max_lon', options.max_lon.toString());
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/generate-heatmap`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error generating heatmap:', error);
    throw error;
  }
}

/**
 * Generate demo heatmap without video input
 */
export async function generateDemoHeatmap(options = {}) {
  const formData = new FormData();
  
  if (options.grid_size !== undefined) {
    formData.append('grid_size', options.grid_size.toString());
  }
  if (options.min_lat !== undefined) {
    formData.append('min_lat', options.min_lat.toString());
  }
  if (options.max_lat !== undefined) {
    formData.append('max_lat', options.max_lat.toString());
  }
  if (options.min_lon !== undefined) {
    formData.append('min_lon', options.min_lon.toString());
  }
  if (options.max_lon !== undefined) {
    formData.append('max_lon', options.max_lon.toString());
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/generate-demo-heatmap`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error generating demo heatmap:', error);
    throw error;
  }
}

/**
 * Check backend health
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return { healthy: false };
    return { healthy: true, ...(await response.json()) };
  } catch (error) {
    return { healthy: false };
  }
}
