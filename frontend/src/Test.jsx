import { useState } from 'react'

function Test() {
  const [count, setCount] = useState(0)

  return (
    <div style={{ 
      padding: '20px', 
      fontFamily: 'Arial', 
      background: '#1a1a2e',
      color: 'white',
      minHeight: '100vh'
    }}>
      <h1>🗺️ Mapathon Frontend Test</h1>
      <p>If you can see this, React is working!</p>
      
      <div style={{ marginTop: '20px', padding: '20px', background: '#0f0f23', borderRadius: '8px' }}>
        <h2>Counter Test</h2>
        <button 
          onClick={() => setCount(count + 1)}
          style={{
            padding: '10px 20px',
            background: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          Count: {count}
        </button>
      </div>

      <div style={{ marginTop: '20px', padding: '20px', background: '#0f0f23', borderRadius: '8px' }}>
        <h2>Backend Connection Test</h2>
        <p>Backend URL: <code>http://localhost:8000</code></p>
        <button 
          onClick={async () => {
            try {
              const response = await fetch('/api/health')
              const data = await response.json()
              alert(`Backend connected! Status: ${data.status}`)
            } catch (err) {
              alert(`Backend error: ${err.message}`)
            }
          }}
          style={{
            padding: '10px 20px',
            background: '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          Test Backend Connection
        </button>
      </div>

      <div style={{ marginTop: '20px', padding: '20px', background: '#0f0f23', borderRadius: '8px' }}>
        <h2>Status</h2>
        <ul style={{ lineHeight: '2' }}>
          <li>✅ React: Working</li>
          <li>✅ Vite Dev Server: Running on port 5175</li>
          <li>✅ Hot Reload: Enabled</li>
          <li>⏳ Backend: Test with button above</li>
        </ul>
      </div>

      <div style={{ marginTop: '20px' }}>
        <a 
          href="/"
          onClick={(e) => {
            e.preventDefault()
            window.location.href = '/'
          }}
          style={{ color: '#2563eb', textDecoration: 'underline' }}
        >
          Back to Main App
        </a>
      </div>
    </div>
  )
}

export default Test
