import { useState } from 'react';
import './PredictionForm.css';

const MODELS = [
  {
    value: 'ensemble',
    label: 'Ensemble',
    tag: 'Recommended',
    description: '78% accuracy - Best overall'
  },
  {
    value: 'svc',
    label: 'SVC Advanced',
    tag: 'Highest',
    description: '78% accuracy - Highest accuracy'
  }
];

function PredictionForm({ onSubmit, isLoading }) {
  const [text, setText] = useState('');
  const [model, setModel] = useState('ensemble');
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (text.trim().length < 10) {
      alert('Please enter at least 10 characters');
      return;
    }

    // compare disabled for deployment stability
    onSubmit(text, model, false);
  };

  const charCount = text.length;

  const isValid =
    charCount >= 10 &&
    charCount <= 5000;

  const progress = Math.min(
    (charCount / 10) * 100,
    100
  );

  return (
    <form
      className="prediction-form"
      onSubmit={handleSubmit}
    >
      <div className="form-header">
        <div className="header-glow"></div>

        <h2 className="form-title">
          Mental Health Analysis
        </h2>

        <p className="form-subtitle">
          Share your thoughts in a safe, confidential space
        </p>
      </div>

      {/* TEXTAREA */}

      <div
        className={`form-group textarea-group ${
          isFocused ? 'focused' : ''
        }`}
      >
        <label
          htmlFor="text-input"
          className="floating-label"
        >
          <span className="label-text">
            <span className="label-icon">✍️</span>
            Express your feelings
          </span>

          <span
            className={`char-counter ${
              isValid ? 'valid' : 'invalid'
            }`}
          >
            <span className="count-number">
              {charCount}
            </span>

            <span className="count-divider">
              /
            </span>

            <span className="count-max">
              5000
            </span>

            {charCount < 10 && (
              <span className="count-hint">
                {' '}
                • min 10 chars
              </span>
            )}
          </span>
        </label>

        <div className="textarea-wrapper">
          <textarea
            id="text-input"
            value={text}
            onChange={(e) =>
              setText(e.target.value)
            }
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="I've been feeling anxious lately and can't stop worrying about everything..."
            rows={6}
            maxLength={5000}
            disabled={isLoading}
          />

          <div className="textarea-border"></div>

          {charCount >= 10 && (
            <div
              className="progress-indicator"
              style={{
                width: `${progress}%`
              }}
            ></div>
          )}
        </div>
      </div>

      {/* MODEL SELECTION */}

      <div className="form-group select-group">
        <label
          htmlFor="model-select"
          className="floating-label"
        >
          <span className="label-text">
            <span className="label-icon">
              🧠
            </span>

            AI Model Selection
          </span>
        </label>

        <div className="select-wrapper">
          <select
            id="model-select"
            value={model}
            onChange={(e) =>
              setModel(e.target.value)
            }
            disabled={isLoading}
          >
            {MODELS.map((m) => (
              <option
                key={m.value}
                value={m.value}
              >
                {m.label} • {m.description}
              </option>
            ))}
          </select>

          <div className="select-arrow">
            <svg
              width="12"
              height="8"
              viewBox="0 0 12 8"
              fill="none"
            >
              <path
                d="M1 1.5L6 6.5L11 1.5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* MODEL CARDS */}

        <div className="model-cards">
          {MODELS.map((m) => (
            <div
              key={m.value}
              className={`model-card ${
                model === m.value
                  ? 'active'
                  : ''
              }`}
              onClick={() =>
                !isLoading &&
                setModel(m.value)
              }
            >
              <div className="card-shine"></div>

              <span className="model-tag">
                {m.tag}
              </span>

              <span className="model-name">
                {m.label}
              </span>

              <span className="model-accuracy">
                {
                  m.description.split(
                    ' - '
                  )[0]
                }
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* SUBMIT */}

      <div className="form-actions">
        <button
          type="submit"
          disabled={!isValid || isLoading}
          className="submit-btn"
        >
          <span className="btn-background"></span>

          <span className="btn-content">
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <span className="btn-icon">
                  ✨
                </span>
                Analyze
              </>
            )}
          </span>
        </button>
      </div>
    </form>
  );
}

export default PredictionForm;