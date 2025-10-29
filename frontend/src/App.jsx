import { useState, useEffect } from 'react';
import PredictionForm from './components/PredictionForm';
import ResultDisplay from './components/ResultDisplay';
import { predict, compareModels, checkHealth } from './services/api';
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCompare, setIsCompare] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'));
  }, []);

  const handleSubmit = async (text, model, showCompare) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setIsCompare(showCompare);

    try {
      let data;
      if (showCompare) {
        data = await compareModels(text);
      } else {
        data = await predict(text, model);
      }
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>🧠 Mental Health AI</h1>
          <p>AI-powered mental health condition detection</p>
          <div className="status-badge" data-status={backendStatus}>
            {backendStatus === 'connected' ? '✓ Connected' : 
             backendStatus === 'disconnected' ? '✗ Backend Offline' : 
             '⟳ Connecting...'}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="container">
          {backendStatus === 'disconnected' && (
            <div className="alert alert-error">
              <strong>Backend Offline</strong>
              <p>Please make sure the backend is running on http://localhost:8000</p>
              <code>cd backend && python main.py</code>
            </div>
          )}

          <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} />

          {isLoading && (
            <div className="loading-container">
              <div className="spinner"></div>
              <p>Analyzing your text...</p>
            </div>
          )}

          {error && (
            <div className="alert alert-error">
              <strong>Error</strong>
              <p>{error}</p>
            </div>
          )}

          {result && <ResultDisplay result={result} isCompare={isCompare} />}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          <strong>Disclaimer:</strong> This tool is for educational purposes only and is not a 
          substitute for professional mental health care.
        </p>
        <p>If you're in crisis, please contact emergency services or a crisis helpline immediately.</p>
      </footer>
    </div>
  );
}

export default App;