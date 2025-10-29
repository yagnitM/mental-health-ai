// API Service for Mental Health AI Backend
const API_BASE_URL = '/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Error [${endpoint}]:`, error);
    throw error;
  }
}

/**
 * Health check
 */
export async function checkHealth() {
  return apiFetch('/health');
}

/**
 * Get all available models
 */
export async function getModels() {
  return apiFetch('/models');
}

/**
 * Get all mental health categories
 */
export async function getCategories() {
  return apiFetch('/categories');
}

/**
 * Make a single prediction
 * @param {string} text - Text to analyze
 * @param {string} model - Model name (default: 'ensemble')
 * @param {number} topK - Number of predictions to return (default: 3)
 */
export async function predict(text, model = 'ensemble', topK = 3) {
  return apiFetch('/predict', {
    method: 'POST',
    body: JSON.stringify({
      text,
      model,
      top_k: topK,
    }),
  });
}

/**
 * Compare predictions across all models
 * @param {string} text - Text to analyze
 */
export async function compareModels(text) {
  return apiFetch('/predict/compare', {
    method: 'POST',
    body: JSON.stringify({
      text,
      model: 'ensemble', // Required but not used in comparison
      top_k: 3,
    }),
  });
}

/**
 * Batch prediction for multiple texts
 * @param {string[]} texts - Array of texts to analyze
 * @param {string} model - Model name (default: 'ensemble')
 */
export async function predictBatch(texts, model = 'ensemble') {
  return apiFetch('/predict/batch', {
    method: 'POST',
    body: JSON.stringify({
      texts,
      model,
    }),
  });
}