
import { useState, useEffect, useRef } from 'react';

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

  // Tracks whether any API request has already succeeded.
  const backendAvailableRef = useRef(false);

  // Check backend health when the app loads.
  useEffect(() => {
    let cancelled = false;

    async function verifyBackend() {
      try {
        await checkHealth();

        if (cancelled) return;

        backendAvailableRef.current = true;
        setBackendStatus('connected');
      } catch (err) {
        console.error('Backend health check failed:', err);

        if (cancelled || backendAvailableRef.current) return;

        setBackendStatus('disconnected');
      }
    }

    verifyBackend();

    return () => {
      cancelled = true;
    };
  }, []);

  // Run an analysis.
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

      // A successful prediction proves that the prediction
      // endpoint is reachable, even if /health previously failed.
      backendAvailableRef.current = true;
      setBackendStatus('connected');
      setResult(data);
    } catch (err) {
      console.error('Analysis request failed:', err);

      setError(
        err.message || 'Failed to analyze your text. Please try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const statusLabels = {
    checking: '⟳ Checking connection...',
    connected: '✓ Connected',
    disconnected: '✗ Connection unavailable',
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🧠 Mental Health AI</h1>
          <p>AI-powered mental health condition detection</p>

          <div
            className="status-badge"
            data-status={backendStatus}
            role="status"
            aria-live="polite"
          >
            {statusLabels[backendStatus]}
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          {backendStatus === 'disconnected' && (
            <div className="alert alert-error" role="status">
              <strong>Connection check failed</strong>
              <p>
                The health endpoint could not be reached. You can still try
                submitting an analysis.
              </p>
            </div>
          )}

          <PredictionForm
            onSubmit={handleSubmit}
            isLoading={isLoading}
          />

          {isLoading && (
            <div className="loading-container" role="status">
              <div className="spinner" />
              <p>Analyzing your text...</p>
            </div>
          )}

          {error && (
            <div className="alert alert-error" role="alert">
              <strong>Analysis failed</strong>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <ResultDisplay
              result={result}
              isCompare={isCompare}
            />
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>
          <strong>Disclaimer:</strong> This tool is for educational
          purposes only and is not a substitute for professional mental
          health care.
        </p>

        <p>
          If you're in crisis, please contact emergency services or a
          crisis helpline immediately.
        </p>
      </footer>
    </div>
  );
}

export default App;
