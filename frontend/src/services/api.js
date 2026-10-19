/**
 * API Service for Mental Health AI Backend
 * Supports local development and production deployment.
 *
 * Local development:
 *   VITE_API_URL=http://localhost:8000
 *
 * Production:
 *   VITE_API_URL=https://your-backend.onrender.com
 */

const API_BASE_URL = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000' : '')
).replace(/\/+$/, '');

// Validate configuration early.
if (!API_BASE_URL) {
  console.error(
    '[API] VITE_API_URL is missing. Configure it in Vercel Environment Variables and redeploy.'
  );
} else {
  console.info('[API] Base URL:', API_BASE_URL);
}

/**
 * Build a useful error message from an API response.
 */
async function getErrorMessage(response) {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    try {
      const data = await response.json();

      if (typeof data.detail === 'string') {
        return data.detail;
      }

      if (typeof data.error === 'string') {
        return data.error;
      }

      if (Array.isArray(data.detail)) {
        return data.detail
          .map((item) => item.msg || JSON.stringify(item))
          .join('; ');
      }

      return JSON.stringify(data);
    } catch {
      // Fall through to the HTTP status message.
    }
  }

  return response.statusText || 'The server returned an error.';
}

/**
 * Generic fetch wrapper.
 */
async function apiFetch(endpoint, options = {}) {
  if (!API_BASE_URL) {
    throw new Error(
      'API URL is not configured. Set VITE_API_URL in your deployment settings.'
    );
  }

  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const message = await getErrorMessage(response);

      throw new Error(
        `${response.status} ${response.statusText}: ${message}`
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      console.error(`[API] Network error: ${url}`, error);

      throw new Error(
        'Could not connect to the backend. Check the backend URL, service availability, and CORS configuration.'
      );
    }

    console.error(`[API] Request failed: ${url}`, error);
    throw error;
  }
}

/**
 * Check backend health.
 */
export async function checkHealth() {
  const health = await apiFetch('/health');

  if (health.status !== 'healthy') {
    throw new Error('The backend responded but is not reporting healthy status.');
  }

  return health;
}

/**
 * Get available models.
 */
export async function getModels() {
  return apiFetch('/models');
}

/**
 * Get mental health categories.
 */
export async function getCategories() {
  return apiFetch('/categories');
}

/**
 * Make a single prediction.
 *
 * @param {string} text - Text to analyze
 * @param {'svc' | 'ensemble'} model - Model to use
 * @param {number} topK - Number of predictions to return
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
 * Compare predictions across models.
 *
 * @param {string} text - Text to analyze
 */
export async function compareModels(text) {
  return apiFetch('/predict/compare', {
    method: 'POST',
    body: JSON.stringify({
      text,
      model: 'ensemble',
      top_k: 3,
    }),
  });
}

/**
 * Run predictions on multiple texts.
 *
 * @param {string[]} texts - Texts to analyze
 * @param {'svc' | 'ensemble'} model - Model to use
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
