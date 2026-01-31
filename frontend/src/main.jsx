import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
// import Test from './Test.jsx'  // Uncomment to use test page
import 'leaflet/dist/leaflet.css'
import './styles/index.css'

// To debug: change App to Test below
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
