import { useState } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setResults(null)
    setError(null)
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV file first.")
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Server error. Ensure the CSV format is correct.")
      }

      const data = await response.json()
      setResults(data.predictions)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ fontFamily: 'Arial', padding: '40px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Exoplanet Detection Portal</h1>
      <p>Upload your Kepler observation data to detect exoplanets.</p>

      <div style={{ padding: '20px', border: '2px dashed #ccc', borderRadius: '8px', marginBottom: '20px' }}>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button 
          onClick={handleUpload} 
          disabled={loading}
          style={{ marginLeft: '10px', padding: '8px 16px', cursor: 'pointer' }}
        >
          {loading ? "Analyzing..." : "Run Detection"}
        </button>
      </div>

      {error && <div style={{ color: 'red', marginBottom: '20px' }}>{error}</div>}

      {results && (
        <div>
          <h2>Analysis Results</h2>
          {results.map((item, index) => (
            <div key={index} style={{ 
              border: '1px solid #ddd', 
              padding: '15px', 
              borderRadius: '8px', 
              marginBottom: '15px',
              backgroundColor: item.prediction === 'CONFIRMED' ? '#e6ffe6' : '#fff'
            }}>
              <h3>Row {item.row_index} {item.kepid && `| Kepler ID: ${item.kepid}`}</h3>
              <p><strong>Verdict: </strong> <span style={{ fontSize: '1.2em' }}>{item.prediction}</span></p>
              
              <h4>Confidence Scores:</h4>
              <ul style={{ listStyleType: 'none', padding: 0 }}>
                <li>CONFIRMED: {item.probabilities["CONFIRMED"]}%</li>
                <li>CANDIDATE: {item.probabilities["CANDIDATE"]}%</li>
                <li>FALSE POSITIVE: {item.probabilities["FALSE POSITIVE"]}%</li>
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App