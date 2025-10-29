"""
Mental Health AI - Complete FastAPI Backend
============================================
This is a production-ready FastAPI application for mental health condition detection.
Place this file in: mental-health-ai/backend/main.py

Directory structure expected:
mental-health-ai/
├── backend/
│   ├── main.py (this file)
│   └── venv/
├── src/
│   └── models/
│       ├── svc_model.pkl
│       ├── tfidf_word.pkl
│       ├── tfidf_char.pkl
│       ├── lr_sbert.pkl
│       ├── baseline_logistic_regression.pkl
│       ├── baseline_random_forest.pkl
│       └── tfidf_vectorizer.pkl
└── data/
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
import joblib
import numpy as np
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
import os
import sys
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="Mental Health AI Detection API",
    description="AI-powered mental health condition detection using ensemble learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Variables & Constants
# ============================================================================

models = {}
LABELS = [
    'addiction', 'adhd', 'anxiety', 'autism', 'bipolar',
    'bpd', 'depression', 'ocd', 'psychosis', 'ptsd', 'suicide'
]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# Model metadata with actual performance metrics
MODEL_INFO = {
    "ensemble": {
        "name": "Ensemble (Weighted)",
        "description": "Weighted combination of all 4 models for best overall performance",
        "accuracy": 0.73,
        "type": "ensemble",
        "weights": {"svc": 0.4, "sbert": 0.3, "lr": 0.15, "rf": 0.15}
    },
    "svc": {
        "name": "Support Vector Classifier",
        "description": "LinearSVC with advanced TF-IDF features (word + char n-grams)",
        "accuracy": 0.78,
        "type": "advanced"
    },
    "sbert_lr": {
        "name": "SBERT + Logistic Regression",
        "description": "Sentence transformers embeddings with LR classifier",
        "accuracy": 0.71,
        "type": "advanced"
    },
    "baseline_lr": {
        "name": "Baseline Logistic Regression",
        "description": "Simple LR with standard TF-IDF features",
        "accuracy": 0.73,
        "type": "baseline"
    },
    "baseline_rf": {
        "name": "Baseline Random Forest",
        "description": "Random Forest with standard TF-IDF features",
        "accuracy": 0.69,
        "type": "baseline"
    }
}

# Category descriptions for user understanding
CATEGORY_DESCRIPTIONS = {
    "addiction": "Substance abuse and addictive behaviors",
    "adhd": "Attention Deficit Hyperactivity Disorder - difficulty focusing and hyperactivity",
    "anxiety": "Anxiety disorders, panic attacks, and excessive worry",
    "autism": "Autism Spectrum Disorder - social and communication challenges",
    "bipolar": "Bipolar disorder - mood swings between mania and depression",
    "bpd": "Borderline Personality Disorder - emotional instability and relationship issues",
    "depression": "Major depressive disorder - persistent sadness and loss of interest",
    "ocd": "Obsessive Compulsive Disorder - intrusive thoughts and repetitive behaviors",
    "psychosis": "Psychotic disorders - loss of contact with reality",
    "ptsd": "Post-Traumatic Stress Disorder - trauma-related symptoms",
    "suicide": "Suicidal ideation - thoughts of self-harm or suicide"
}

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class TextInput(BaseModel):
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="Text to analyze for mental health conditions",
        example="I've been feeling really anxious lately and can't stop worrying"
    )
    model: Literal["ensemble", "svc", "sbert_lr", "baseline_lr", "baseline_rf"] = Field(
        default="ensemble",
        description="Model to use for prediction"
    )
    top_k: int = Field(
        default=3, 
        ge=1, 
        le=11,
        description="Number of top predictions to return"
    )

class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ..., 
        max_items=50,
        description="List of texts to analyze (max 50)"
    )
    model: Literal["ensemble", "svc", "sbert_lr", "baseline_lr", "baseline_rf"] = Field(
        default="ensemble"
    )
    top_k: int = Field(default=3, ge=1, le=11)

class PredictionResult(BaseModel):
    category: str
    confidence: float

class PredictionResponse(BaseModel):
    text: str
    model_used: str
    predictions: List[PredictionResult]
    top_prediction: str
    confidence: float
    model_info: Dict

class ModelComparisonResponse(BaseModel):
    text: str
    results: Dict[str, Dict]

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict
    categories: List[str]
    total_models: int

# ============================================================================
# Model Loading
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load all ML models and vectorizers on startup"""
    global models
    
    try:
        # Determine model directory path
        current_file = Path(__file__).resolve()
        backend_dir = current_file.parent
        project_root = backend_dir.parent
        MODEL_DIR = project_root / "src" / "models"
        
        print("\n" + "="*60)
        print("🚀 Starting Mental Health AI Backend")
        print("="*60)
        print(f"📂 Backend directory: {backend_dir}")
        print(f"📂 Project root: {project_root}")
        print(f"📂 Model directory: {MODEL_DIR}")
        print("="*60 + "\n")
        
        # Check if model directory exists
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory not found: {MODEL_DIR}\n"
                f"Please ensure models are trained and saved in {MODEL_DIR}"
            )
        
        # Required model files
        required_files = [
            "svc_model.pkl",
            "tfidf_word.pkl",
            "tfidf_char.pkl",
            "lr_sbert.pkl",
            "baseline_logistic_regression.pkl",
            "baseline_random_forest.pkl",
            "tfidf_vectorizer.pkl"
        ]
        
        # Check all required files exist
        missing_files = []
        for filename in required_files:
            filepath = MODEL_DIR / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing_files)}\n"
                f"Please train models by running:\n"
                f"  cd {project_root / 'src' / 'models'}\n"
                f"  python train_baseline_model.py\n"
                f"  python train_advanced_model.py"
            )
        
        print("📥 Loading models...")
        
        # Load advanced models (SVC)
        print("   Loading SVC model...")
        models['svc'] = joblib.load(MODEL_DIR / "svc_model.pkl")
        models['tfidf_word'] = joblib.load(MODEL_DIR / "tfidf_word.pkl")
        models['tfidf_char'] = joblib.load(MODEL_DIR / "tfidf_char.pkl")
        
        # Verify vectorizers are fitted
        if not hasattr(models['tfidf_word'], 'vocabulary_'):
            raise ValueError("TF-IDF word vectorizer is not fitted!")
        if not hasattr(models['tfidf_char'], 'vocabulary_'):
            raise ValueError("TF-IDF char vectorizer is not fitted!")
        
        print(f"      ✅ Word vocab: {len(models['tfidf_word'].vocabulary_)} features")
        print(f"      ✅ Char vocab: {len(models['tfidf_char'].vocabulary_)} features")
        
        # Load SBERT model
        print("   Loading SBERT-LR model...")
        models['lr_sbert'] = joblib.load(MODEL_DIR / "lr_sbert.pkl")
        
        # Load baseline models
        print("   Loading baseline models...")
        models['baseline_lr'] = joblib.load(MODEL_DIR / "baseline_logistic_regression.pkl")
        models['baseline_rf'] = joblib.load(MODEL_DIR / "baseline_random_forest.pkl")
        models['baseline_vectorizer'] = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
        
        # Verify baseline vectorizer
        if not hasattr(models['baseline_vectorizer'], 'vocabulary_'):
            raise ValueError("Baseline TF-IDF vectorizer is not fitted!")
        print(f"      ✅ Baseline vocab: {len(models['baseline_vectorizer'].vocabulary_)} features")
        
        # Load SBERT transformer
        print("   Loading SBERT transformer (this may take a moment)...")
        models['sbert_model'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("\n" + "="*60)
        print("✅ All models loaded successfully!")
        print("="*60)
        print(f"   🎯 SVC (Advanced)          - 78% accuracy")
        print(f"   🤖 SBERT-LR (Advanced)     - 71% accuracy")
        print(f"   📊 Baseline LR             - 73% accuracy")
        print(f"   🌳 Baseline RF             - 69% accuracy")
        print(f"   🎭 Ensemble (Weighted)     - 73% accuracy")
        print("="*60)
        print(f"📋 Categories: {', '.join(LABELS)}")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Backend will start but predictions will fail!")
        print("    Please train models first.\n")
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print(f"\n📋 Traceback:")
        traceback.print_exc()
        print("\n⚠️  Backend will start but predictions will fail!\n")

# ============================================================================
# Feature Preparation
# ============================================================================

def prepare_features(texts: List[str]) -> Dict:
    """
    Prepare features for all models from input texts
    
    Args:
        texts: List of text strings to process
        
    Returns:
        Dictionary containing features for each model type
    """
    if not models:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please restart the server."
        )
    
    try:
        # Advanced TF-IDF features (word + char) for SVC
        X_word = models['tfidf_word'].transform(texts)
        X_char = models['tfidf_char'].transform(texts)
        X_advanced = hstack([X_word, X_char])
        
        # SBERT embeddings for SBERT-LR
        embeddings = models['sbert_model'].encode(
            texts, 
            batch_size=32, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        
        # Baseline TF-IDF for baseline models
        X_baseline = models['baseline_vectorizer'].transform(texts)
        
        return {
            'advanced': X_advanced,
            'sbert': embeddings,
            'baseline': X_baseline
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error preparing features: {str(e)}"
        )

# ============================================================================
# Prediction Functions
# ============================================================================

def predict_with_model(
    features: Dict, 
    texts: List[str], 
    model_name: str, 
    top_k: int = 3
) -> List[Dict]:
    """
    Make predictions using specified model
    
    Args:
        features: Pre-computed features dictionary
        texts: Original text inputs
        model_name: Name of model to use
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction dictionaries
    """
    try:
        # Get probabilities from selected model
        if model_name == "svc":
            probs = models['svc'].predict_proba(features['advanced'])
            
        elif model_name == "sbert_lr":
            probs = models['lr_sbert'].predict_proba(features['sbert'])
            
        elif model_name == "baseline_lr":
            probs = models['baseline_lr'].predict_proba(features['baseline'])
            
        elif model_name == "baseline_rf":
            probs = models['baseline_rf'].predict_proba(features['baseline'])
            
        elif model_name == "ensemble":
            # Weighted ensemble based on validation performance
            weights = MODEL_INFO['ensemble']['weights']
            
            svc_probs = models['svc'].predict_proba(features['advanced'])
            sbert_probs = models['lr_sbert'].predict_proba(features['sbert'])
            lr_probs = models['baseline_lr'].predict_proba(features['baseline'])
            rf_probs = models['baseline_rf'].predict_proba(features['baseline'])
            
            probs = (
                weights['svc'] * svc_probs +
                weights['sbert'] * sbert_probs +
                weights['lr'] * lr_probs +
                weights['rf'] * rf_probs
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Process results for each text
        results = []
        for i, text in enumerate(texts):
            final_probs = probs[i]
            top_indices = np.argsort(final_probs)[::-1][:top_k]
            
            predictions = [
                {
                    "category": ID2LABEL[idx],
                    "confidence": float(final_probs[idx])
                }
                for idx in top_indices
            ]
            
            results.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'model_used': model_name,
                'predictions': predictions,
                'top_prediction': predictions[0]['category'],
                'confidence': predictions[0]['confidence'],
                'model_info': MODEL_INFO[model_name]
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API status and basic info"""
    return {
        "status": "active",
        "service": "Mental Health AI Detection API",
        "version": "2.0.0",
        "models_loaded": len(models) > 0,
        "available_categories": LABELS,
        "available_models": list(MODEL_INFO.keys()),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "compare": "/predict/compare",
            "models": "/models",
            "categories": "/categories"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with model status"""
    return {
        "status": "healthy" if len(models) > 0 else "unhealthy",
        "models_loaded": {
            "svc": "svc" in models,
            "sbert_lr": "lr_sbert" in models,
            "baseline_lr": "baseline_lr" in models,
            "baseline_rf": "baseline_rf" in models,
            "sbert_model": "sbert_model" in models,
            "vectorizers": all(k in models for k in [
                'tfidf_word', 'tfidf_char', 'baseline_vectorizer'
            ])
        },
        "categories": LABELS,
        "total_models": len(models)
    }

@app.get("/models")
async def get_models():
    """Get information about all available models"""
    return {
        "models": MODEL_INFO,
        "total": len(MODEL_INFO),
        "recommendation": "Use 'ensemble' for best overall performance or 'svc' for highest accuracy",
        "available_models": list(MODEL_INFO.keys())
    }

@app.get("/categories")
async def get_categories():
    """Get all mental health categories with descriptions"""
    return {
        "categories": LABELS,
        "count": len(LABELS),
        "descriptions": CATEGORY_DESCRIPTIONS
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput):
    """
    Predict mental health condition from text using selected model
    
    - **text**: The text to analyze (10-5000 characters)
    - **model**: Model to use (default: ensemble)
    - **top_k**: Number of predictions to return (default: 3)
    """
    if not models:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please restart the server or train models."
        )
    
    try:
        features = prepare_features([input_data.text])
        results = predict_with_model(
            features, 
            [input_data.text], 
            input_data.model, 
            input_data.top_k
        )
        return results[0]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(input_data: BatchTextInput):
    """
    Batch prediction for multiple texts (max 50)
    
    Returns predictions for all texts in a single request
    """
    if not models:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded"
        )
    
    if len(input_data.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 texts allowed per batch"
        )
    
    try:
        features = prepare_features(input_data.texts)
        results = predict_with_model(
            features, 
            input_data.texts, 
            input_data.model,
            input_data.top_k
        )
        return {
            "predictions": results, 
            "count": len(results),
            "model_used": input_data.model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/predict/compare", response_model=ModelComparisonResponse)
async def compare_models(input_data: TextInput):
    """
    Compare predictions across all 5 models for the same text
    
    Useful for understanding how different models interpret the same input
    """
    if not models:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded"
        )
    
    try:
        text = input_data.text
        features = prepare_features([text])
        results = {}
        
        for model_name in MODEL_INFO.keys():
            prediction = predict_with_model(
                features, 
                [text], 
                model_name, 
                top_k=3
            )[0]
            
            results[model_name] = {
                "top_prediction": prediction['top_prediction'],
                "confidence": prediction['confidence'],
                "top_3_predictions": prediction['predictions'],
                "model_accuracy": MODEL_INFO[model_name]['accuracy'],
                "model_type": MODEL_INFO[model_name]['type']
            }
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model comparison failed: {str(e)}"
        )

# ============================================================================
# Error Handlers
# ============================================================================

from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/", "/health", "/models", "/categories", 
                "/predict", "/predict/batch", "/predict/compare"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Something went wrong. Please check server logs.",
            "suggestion": "If models are not loaded, try retraining them."
        }
    )

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Starting Mental Health AI Backend")
    print("="*60)
    print("📍 Server will be available at:")
    print("   - Local:   http://localhost:8000")
    print("   - Network: http://0.0.0.0:8000")
    print("   - Docs:    http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )