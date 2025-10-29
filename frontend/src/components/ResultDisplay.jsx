import './ResultDisplay.css';

const CATEGORY_INFO = {
  addiction: { emoji: '🚬', color: '#e74c3c' },
  adhd: { emoji: '⚡', color: '#f39c12' },
  anxiety: { emoji: '😰', color: '#e67e22' },
  autism: { emoji: '🧩', color: '#9b59b6' },
  bipolar: { emoji: '🎭', color: '#3498db' },
  bpd: { emoji: '💔', color: '#e91e63' },
  depression: { emoji: '😔', color: '#34495e' },
  ocd: { emoji: '🔄', color: '#16a085' },
  psychosis: { emoji: '🌀', color: '#8e44ad' },
  ptsd: { emoji: '💥', color: '#c0392b' },
  suicide: { emoji: '⚠️', color: '#d32f2f' },
};

function ResultDisplay({ result, isCompare }) {
  if (!result) return null;

  if (isCompare) {
    return (
      <div className="result-container">
        <div className="result-header">
          <h2>Model Comparison Results</h2>
          <p className="analyzed-text">"{result.text}"</p>
        </div>

        <div className="comparison-grid">
          {Object.entries(result.results).map(([modelName, data]) => (
            <div key={modelName} className="comparison-card">
              <div className="model-name">{modelName.toUpperCase()}</div>
              <div className="model-accuracy">{(data.model_accuracy * 100).toFixed(0)}% accuracy</div>
              
              <div className="top-prediction">
                <span className="category-emoji">
                  {CATEGORY_INFO[data.top_prediction]?.emoji || '🔮'}
                </span>
                <div>
                  <div className="category-name">{data.top_prediction.toUpperCase()}</div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${data.confidence * 100}%`,
                        backgroundColor: CATEGORY_INFO[data.top_prediction]?.color || '#3498db',
                      }}
                    />
                  </div>
                  <div className="confidence-text">{(data.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="result-container">
      <div className="result-header">
        <h2>Analysis Results</h2>
        <p className="analyzed-text">"{result.text}"</p>
        <div className="model-badge">
          Model: {result.model_info.name} ({(result.model_info.accuracy * 100).toFixed(0)}% accuracy)
        </div>
      </div>

      <div className="predictions-list">
        {result.predictions.map((pred, index) => {
          const info = CATEGORY_INFO[pred.category] || { emoji: '🔮', color: '#3498db' };
          const percentage = (pred.confidence * 100).toFixed(1);

          return (
            <div key={index} className="prediction-item" data-rank={index === 0 ? 'top' : ''}>
              <div className="prediction-header">
                <span className="category-emoji">{info.emoji}</span>
                <span className="category-name">{pred.category.toUpperCase()}</span>
                <span className="confidence-badge">{percentage}%</span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: info.color,
                  }}
                />
              </div>
              {index === 0 && (
                <div className="top-label">Most Likely</div>
              )}
            </div>
          );
        })}
      </div>

      <div className="disclaimer">
        <strong>⚠️ Important:</strong> This is an AI prediction tool and not a medical diagnosis. 
        Please consult a mental health professional for proper evaluation and treatment.
      </div>
    </div>
  );
}

export default ResultDisplay;