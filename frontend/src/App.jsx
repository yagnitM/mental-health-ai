import { useState } from 'react';

import PredictionForm from './components/PredictionForm';
import ResultDisplay from './components/ResultDisplay';
import { predict, compareModels } from './services/api';

import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCompare, setIsCompare] = useState(false);

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
      console.error('Analysis request failed:', err);

      setError(
        err.message || 'Failed to analyze your text. Please try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🧠 Mental Health AI</h1>
          <p>AI-powered mental health condition detection</p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
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