import { useState } from 'react';
import './PredictionForm.css';

const MODELS = [
  { value: 'ensemble', label: 'Ensemble (Recommended)', description: '73% accuracy - Best overall' },
  { value: 'svc', label: 'SVC Advanced', description: '78% accuracy - Highest accuracy' },
  { value: 'sbert_lr', label: 'SBERT + LR', description: '71% accuracy - Semantic analysis' },
  { value: 'baseline_lr', label: 'Baseline LR', description: '73% accuracy - Fast & reliable' },
  { value: 'baseline_rf', label: 'Baseline RF', description: '69% accuracy - Tree-based' },
];

function PredictionForm({ onSubmit, isLoading }) {
  const [text, setText] = useState('');
  const [model, setModel] = useState('ensemble');
  const [showCompare, setShowCompare] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim().length < 10) {
      alert('Please enter at least 10 characters');
      return;
    }
    onSubmit(text, model, showCompare);
  };

  const charCount = text.length;
  const isValid = charCount >= 10 && charCount <= 5000;

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="text-input">
          Describe your feelings or thoughts
          <span className="char-counter" data-valid={isValid}>
            {charCount}/5000 {charCount < 10 && '(min 10 characters)'}
          </span>
        </label>
        <textarea
          id="text-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Example: I've been feeling really anxious lately and can't stop worrying about everything..."
          rows={6}
          maxLength={5000}
          disabled={isLoading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="model-select">Select AI Model</label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={isLoading}
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label} - {m.description}
            </option>
          ))}
        </select>
      </div>

      <div className="form-actions">
        <label className="compare-checkbox">
          <input
            type="checkbox"
            checked={showCompare}
            onChange={(e) => setShowCompare(e.target.checked)}
            disabled={isLoading}
          />
          <span>Compare all models</span>
        </label>

        <button type="submit" disabled={!isValid || isLoading} className="submit-btn">
          {isLoading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
    </form>
  );
}

export default PredictionForm;