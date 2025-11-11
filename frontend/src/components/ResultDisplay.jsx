import './ResultDisplay.css';

const CATEGORY_INFO = {
  addiction: { emoji: '🚬', color: '#e74c3c', gradient: 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)' },
  adhd: { emoji: '⚡', color: '#f39c12', gradient: 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)' },
  anxiety: { emoji: '😰', color: '#e67e22', gradient: 'linear-gradient(135deg, #e67e22 0%, #d35400 100%)' },
  autism: { emoji: '🧩', color: '#9b59b6', gradient: 'linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)' },
  bipolar: { emoji: '🎭', color: '#3498db', gradient: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)' },
  bpd: { emoji: '💔', color: '#e91e63', gradient: 'linear-gradient(135deg, #e91e63 0%, #c2185b 100%)' },
  depression: { emoji: '😔', color: '#34495e', gradient: 'linear-gradient(135deg, #34495e 0%, #2c3e50 100%)' },
  ocd: { emoji: '🔄', color: '#16a085', gradient: 'linear-gradient(135deg, #16a085 0%, #138d75 100%)' },
  psychosis: { emoji: '🌀', color: '#8e44ad', gradient: 'linear-gradient(135deg, #8e44ad 0%, #71368a 100%)' },
  ptsd: { emoji: '💥', color: '#c0392b', gradient: 'linear-gradient(135deg, #c0392b 0%, #a93226 100%)' },
  suicide: { emoji: '⚠️', color: '#d32f2f', gradient: 'linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%)' },
};

function ResultDisplay({ result, isCompare }) {
  if (!result) return null;

  if (isCompare) {
    return (
      <div className="result-container comparison-mode">
        <div className="result-header">
          <div className="header-ornament">
            <div className="ornament-particle"></div>
            <div className="ornament-particle"></div>
            <div className="ornament-particle"></div>
          </div>
          <h2 className="result-title">
            <span className="title-icon">📊</span>
            Model Comparison Results
          </h2>
          <div className="analyzed-text-wrapper">
            <div className="quote-mark left">"</div>
            <p className="analyzed-text">{result.text}</p>
            <div className="quote-mark right">"</div>
          </div>
        </div>

        <div className="comparison-grid">
          {Object.entries(result.results).map(([modelName, data], index) => {
            const info = CATEGORY_INFO[data.top_prediction] || { emoji: '🔮', color: '#3498db', gradient: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)' };
            return (
              <div key={modelName} className="comparison-card" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="card-glass-overlay"></div>
                <div className="card-content">
                  <div className="model-header">
                    <div className="model-name">{modelName.toUpperCase()}</div>
                    <div className="model-accuracy">
                      <span className="accuracy-value">{(data.model_accuracy * 100).toFixed(0)}</span>
                      <span className="accuracy-unit">%</span>
                    </div>
                  </div>
                  
                  <div className="top-prediction">
                    <div className="prediction-emoji-wrapper">
                      <span className="category-emoji">{info.emoji}</span>
                      <div className="emoji-glow" style={{ background: info.gradient }}></div>
                    </div>
                    <div className="category-name">{data.top_prediction.toUpperCase()}</div>
                    
                    <div className="confidence-visualization">
                      <svg className="confidence-circle" viewBox="0 0 120 120">
                        <circle
                          className="circle-bg"
                          cx="60"
                          cy="60"
                          r="52"
                        />
                        <circle
                          className="circle-progress"
                          cx="60"
                          cy="60"
                          r="52"
                          style={{
                            stroke: info.color,
                            strokeDashoffset: 327 - (327 * data.confidence)
                          }}
                        />
                      </svg>
                      <div className="confidence-text">
                        <span className="confidence-value">{(data.confidence * 100).toFixed(1)}</span>
                        <span className="confidence-percent">%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="disclaimer">
          <div className="disclaimer-icon">⚠️</div>
          <div className="disclaimer-content">
            <strong>Important Notice</strong>
            <p>This is an AI prediction tool and not a medical diagnosis. Please consult a mental health professional for proper evaluation and treatment.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="result-container single-mode">
      <div className="result-header">
        <div className="header-ornament">
          <div className="ornament-particle"></div>
          <div className="ornament-particle"></div>
          <div className="ornament-particle"></div>
        </div>
        <h2 className="result-title">
          <span className="title-icon">🎯</span>
          Analysis Results
        </h2>
        <div className="analyzed-text-wrapper">
          <div className="quote-mark left">"</div>
          <p className="analyzed-text">{result.text}</p>
          <div className="quote-mark right">"</div>
        </div>
        <div className="model-badge">
          <span className="badge-icon">🧠</span>
          <span className="badge-text">
            Model: {result.model_info.name}
          </span>
          <span className="badge-accuracy">
            {(result.model_info.accuracy * 100).toFixed(0)}% accuracy
          </span>
        </div>
      </div>

      <div className="predictions-list">
        {result.predictions.map((pred, index) => {
          const info = CATEGORY_INFO[pred.category] || { emoji: '🔮', color: '#3498db', gradient: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)' };
          const percentage = (pred.confidence * 100).toFixed(1);
          const isTop = index === 0;

          return (
            <div 
              key={index} 
              className={`prediction-item ${isTop ? 'top-prediction' : ''}`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {isTop && (
                <>
                  <div className="top-badge">
                    <span className="badge-star">⭐</span>
                    Most Likely
                  </div>
                  <div className="top-glow" style={{ background: info.gradient }}></div>
                </>
              )}
              
              <div className="prediction-content">
                <div className="prediction-header">
                  <div className="emoji-container">
                    <span className="category-emoji">{info.emoji}</span>
                    {isTop && <div className="emoji-pulse" style={{ borderColor: info.color }}></div>}
                  </div>
                  
                  <div className="category-info">
                    <span className="category-name">{pred.category.toUpperCase()}</span>
                    <span className="category-rank">Rank #{index + 1}</span>
                  </div>
                  
                  <div className="confidence-badge" style={{ background: info.gradient }}>
                    <span className="confidence-value">{percentage}</span>
                    <span className="confidence-unit">%</span>
                  </div>
                </div>
                
                <div className="confidence-bar-container">
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${percentage}%`,
                        background: info.gradient,
                      }}
                    >
                      <div className="fill-shine"></div>
                    </div>
                  </div>
                  <div className="confidence-markers">
                    {[25, 50, 75].map(mark => (
                      <div key={mark} className="marker" style={{ left: `${mark}%` }}></div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="disclaimer">
        <div className="disclaimer-icon">⚠️</div>
        <div className="disclaimer-content">
          <strong>Important Notice</strong>
          <p>This is an AI prediction tool and not a medical diagnosis. Please consult a mental health professional for proper evaluation and treatment.</p>
        </div>
      </div>
    </div>
  );
}

export default ResultDisplay;