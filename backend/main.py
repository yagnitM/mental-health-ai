from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
import joblib
import numpy as np
from scipy.sparse import hstack
import os
import sys
import warnings
import traceback
from pathlib import Path
import shap
import re

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Mental Health AI Detection API",
    description="AI-powered mental health condition detection using ensemble learning with SHAP explainability",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://mental-health-ai-bice.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches - models loaded on first use
models = {}
explainers = {}
_model_dir = None

LABELS = [
    'addiction', 'adhd', 'anxiety', 'autism', 'bipolar',
    'bpd', 'depression', 'ocd', 'psychosis', 'ptsd', 'suicide'
]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

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

class ExplainInput(BaseModel):
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="Text to explain"
    )
    model: Literal["svc", "baseline_lr", "baseline_rf"] = Field(
        default="svc",
        description="Model to use for explanation (SHAP not supported for sbert_lr and ensemble)"
    )
    max_display: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum number of features to display in explanation"
    )

class PredictionResult(BaseModel):
    category: str
    confidence: float

class FeatureImportance(BaseModel):
    token: str
    importance: float
    position: int

class ExplanationResponse(BaseModel):
    text: str
    model_used: str
    predicted_category: str
    confidence: float
    base_value: float
    feature_importances: List[FeatureImportance]
    explanation_type: str

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
    explainers_loaded: bool

# LAZY LOADING FUNCTIONS
def get_model_dir():
    global _model_dir
    if _model_dir is None:
        current_file = Path(__file__).resolve()
        backend_dir = current_file.parent
        _model_dir = backend_dir / "models"
        
        if not _model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {_model_dir}")
    
    return _model_dir

def load_model_lazy(model_key: str):
    """Load a model only when needed"""
    if model_key in models:
        return models[model_key]
    
    MODEL_DIR = get_model_dir()
    
    print(f"   Loading {model_key}...")
    
    if model_key == 'svc':
        models['svc'] = joblib.load(MODEL_DIR / "svc_model.pkl")
    elif model_key == 'tfidf_word':
        models['tfidf_word'] = joblib.load(MODEL_DIR / "tfidf_word.pkl")
    elif model_key == 'tfidf_char':
        models['tfidf_char'] = joblib.load(MODEL_DIR / "tfidf_char.pkl")
    elif model_key == 'lr_sbert':
        models['lr_sbert'] = joblib.load(MODEL_DIR / "lr_sbert.pkl")
    elif model_key == 'baseline_lr':
        models['baseline_lr'] = joblib.load(MODEL_DIR / "baseline_logistic_regression.pkl")
    elif model_key == 'baseline_rf':
        models['baseline_rf'] = joblib.load(MODEL_DIR / "baseline_random_forest.pkl")
    elif model_key == 'baseline_vectorizer':
        models['baseline_vectorizer'] = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    elif model_key == 'sbert_model':
        # Only load SBERT when actually needed
        from sentence_transformers import SentenceTransformer
        models['sbert_model'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"      ✅ {model_key} loaded")
    return models[model_key]

def load_explainer_lazy(model_name: str):
    """Load a SHAP explainer only when needed"""
    if model_name in explainers:
        return explainers[model_name]
    
    print(f"   Initializing {model_name} explainer...")
    
    if model_name == 'svc':
        load_model_lazy('svc')
        load_model_lazy('tfidf_word')
        load_model_lazy('tfidf_char')
        
        def predict_fn(texts):
            X_word = models['tfidf_word'].transform(texts)
            X_char = models['tfidf_char'].transform(texts)
            X_combined = hstack([X_word, X_char])
            return models['svc'].predict_proba(X_combined)
        
        masker = shap.maskers.Text(tokenizer=r"\W+")
        explainers['svc'] = shap.Explainer(predict_fn, masker, output_names=LABELS)
        
    elif model_name == 'baseline_lr':
        load_model_lazy('baseline_lr')
        load_model_lazy('baseline_vectorizer')
        
        def predict_fn(texts):
            X = models['baseline_vectorizer'].transform(texts)
            return models['baseline_lr'].predict_proba(X)
        
        masker = shap.maskers.Text(tokenizer=r"\W+")
        explainers['baseline_lr'] = shap.Explainer(predict_fn, masker, output_names=LABELS)
        
    elif model_name == 'baseline_rf':
        load_model_lazy('baseline_rf')
        load_model_lazy('baseline_vectorizer')
        
        def predict_fn(texts):
            X = models['baseline_vectorizer'].transform(texts)
            return models['baseline_rf'].predict_proba(X)
        
        masker = shap.maskers.Text(tokenizer=r"\W+")
        explainers['baseline_rf'] = shap.Explainer(predict_fn, masker, output_names=LABELS)
    
    print(f"      ✅ {model_name} explainer ready")
    return explainers[model_name]

@app.on_event("startup")
async def startup():
    """Minimal startup - just verify model directory exists"""
    global _model_dir
    
    print("\n" + "="*60)
    print("🚀 Starting Mental Health AI Backend (Lazy Loading Mode)")
    print("="*60)
    
    try:
        MODEL_DIR = get_model_dir()
        print(f"📂 Model directory: {MODEL_DIR}")
        print("✅ Models will be loaded on first use")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("⚠️  Backend will start but predictions will fail!\n")

def prepare_features(texts: List[str], model_name: str) -> Dict:
    """Prepare features, loading only required models"""
    try:
        features = {}
        
        # Load only what's needed for the requested model
        if model_name in ['svc', 'ensemble']:
            load_model_lazy('tfidf_word')
            load_model_lazy('tfidf_char')
            X_word = models['tfidf_word'].transform(texts)
            X_char = models['tfidf_char'].transform(texts)
            features['advanced'] = hstack([X_word, X_char])
        
        if model_name in ['sbert_lr', 'ensemble']:
            load_model_lazy('sbert_model')
            features['sbert'] = models['sbert_model'].encode(
                texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
            )
        
        if model_name in ['baseline_lr', 'baseline_rf', 'ensemble']:
            load_model_lazy('baseline_vectorizer')
            features['baseline'] = models['baseline_vectorizer'].transform(texts)
        
        return features
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error preparing features: {str(e)}"
        )

def predict_with_model(
    features: Dict, 
    texts: List[str], 
    model_name: str, 
    top_k: int = 3
) -> List[Dict]:
    try:
        if model_name == "svc":
            load_model_lazy('svc')
            probs = models['svc'].predict_proba(features['advanced'])
            
        elif model_name == "sbert_lr":
            load_model_lazy('lr_sbert')
            probs = models['lr_sbert'].predict_proba(features['sbert'])
            
        elif model_name == "baseline_lr":
            load_model_lazy('baseline_lr')
            probs = models['baseline_lr'].predict_proba(features['baseline'])
            
        elif model_name == "baseline_rf":
            load_model_lazy('baseline_rf')
            probs = models['baseline_rf'].predict_proba(features['baseline'])
            
        elif model_name == "ensemble":
            weights = MODEL_INFO['ensemble']['weights']
            
            load_model_lazy('svc')
            load_model_lazy('lr_sbert')
            load_model_lazy('baseline_lr')
            load_model_lazy('baseline_rf')
            
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

@app.get("/")
async def root():
    return {
        "status": "active",
        "service": "Mental Health AI Detection API with SHAP Explainability",
        "version": "2.0.0",
        "models_loaded": len(models),
        "explainers_loaded": len(explainers),
        "available_categories": LABELS,
        "available_models": list(MODEL_INFO.keys()),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "compare": "/predict/compare",
            "explain": "/explain",
            "models": "/models",
            "categories": "/categories"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
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
        "total_models": len(models),
        "explainers_loaded": len(explainers) > 0
    }

@app.get("/models")
async def get_models():
    return {
        "models": MODEL_INFO,
        "total": len(MODEL_INFO),
        "recommendation": "Use 'ensemble' for best overall performance or 'svc' for highest accuracy",
        "available_models": list(MODEL_INFO.keys()),
        "explainable_models": ["svc", "baseline_lr", "baseline_rf"]
    }

@app.get("/categories")
async def get_categories():
    return {
        "categories": LABELS,
        "count": len(LABELS),
        "descriptions": CATEGORY_DESCRIPTIONS
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput):
    try:
        features = prepare_features([input_data.text], input_data.model)
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
    if len(input_data.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 texts allowed per batch"
        )
    
    try:
        features = prepare_features(input_data.texts, input_data.model)
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
    try:
        text = input_data.text
        results = {}
        
        for model_name in MODEL_INFO.keys():
            features = prepare_features([text], model_name)
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

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(input_data: ExplainInput):
    try:
        text = input_data.text
        explainer = load_explainer_lazy(input_data.model)
        
        features = prepare_features([text], input_data.model)
        results = predict_with_model(features, [text], input_data.model, top_k=1)
        predicted_category = results[0]['top_prediction']
        confidence = results[0]['confidence']
        
        print(f"Computing SHAP values for: {text[:50]}...")
        shap_values = explainer([text])
        
        predicted_idx = LABEL2ID[predicted_category]
        
        tokens = re.split(r"\W+", text.lower())
        tokens = [t for t in tokens if t]
        
        token_shap_values = shap_values[0][:, predicted_idx].values
        base_value = float(shap_values[0][:, predicted_idx].base_values)
        
        feature_importances = []
        for i, (token, importance) in enumerate(zip(tokens, token_shap_values)):
            if i >= input_data.max_display:
                break
            feature_importances.append({
                "token": token,
                "importance": float(importance),
                "position": i
            })
        
        feature_importances.sort(key=lambda x: abs(x['importance']), reverse=True)
        feature_importances = feature_importances[:input_data.max_display]
        
        return {
            "text": text,
            "model_used": input_data.model,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "base_value": base_value,
            "feature_importances": feature_importances,
            "explanation_type": "SHAP (Shapley Additive Explanations)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}\n{traceback.format_exc()}"
        )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/", "/health", "/models", "/categories", 
                "/predict", "/predict/batch", "/predict/compare", "/explain"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Something went wrong. Please check server logs.",
            "suggestion": "If models are not loaded, try retraining them."
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Starting Mental Health AI Backend with SHAP Explainability")
    print("="*60)
    print("📍 Server will be available at:")
    print("   - Local:   http://localhost:8000")
    print("   - Network: http://0.0.0.0:8000")
    print("   - Docs:    http://localhost:8000/docs")
    print("   - Explain: http://localhost:8000/explain")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )