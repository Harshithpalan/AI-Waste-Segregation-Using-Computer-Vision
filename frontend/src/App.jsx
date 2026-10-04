import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const handleClassify = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('image', selectedFile)

    try {
      const response = await fetch('http://localhost:5000/api/classify', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Classification failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to classify image')
    } finally {
      setLoading(false)
    }
  }

  const getCategoryColor = (category) => {
    const colors = {
      'recyclable': '#4CAF50',
      'organic': '#8BC34A',
      'hazardous': '#F44336',
      'non-recyclable': '#FF9800'
    }
    return colors[category] || '#2196F3'
  }

  const getCategoryIcon = (category) => {
    const icons = {
      'recyclable': '♻️',
      'organic': '🍃',
      'hazardous': '⚠️',
      'non-recyclable': '🗑️'
    }
    return icons[category] || '📦'
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🌍 AI Waste Segregation</h1>
        <p>Classify waste items using computer vision</p>
      </header>

      <main className="main">
        <div className="upload-section">
          <div className="upload-area">
            {preview ? (
              <img src={preview} alt="Preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon">📷</span>
                <p>Upload an image of waste</p>
              </div>
            )}
          </div>

          <input
            type="file"
            id="file-input"
            accept="image/*"
            onChange={handleFileSelect}
            className="file-input"
          />
          <label htmlFor="file-input" className="upload-button">
            Choose Image
          </label>

          {selectedFile && (
            <button
              onClick={handleClassify}
              disabled={loading}
              className="classify-button"
            >
              {loading ? 'Classifying...' : 'Classify Waste'}
            </button>
          )}
        </div>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>Classification Result</h2>
            <div
              className="result-card"
              style={{ backgroundColor: getCategoryColor(result.category) }}
            >
              <div className="result-icon">
                {getCategoryIcon(result.category)}
              </div>
              <div className="result-content">
                <h3 className="result-category">
                  {result.category.charAt(0).toUpperCase() + result.category.slice(1)}
                </h3>
                <p className="result-confidence">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="categories-info">
              <h3>All Categories</h3>
              <div className="categories-list">
                {result.all_categories.map((cat) => (
                  <div key={cat} className="category-item">
                    <span className="category-icon">{getCategoryIcon(cat)}</span>
                    <span className="category-name">
                      {cat.charAt(0).toUpperCase() + cat.slice(1)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Built with React + Flask + TensorFlow</p>
      </footer>
    </div>
  )
}

export default App
