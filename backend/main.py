from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
import joblib
import numpy as np
from scipy.sparse import hstack
from pathlib import Path
import traceback
import warnings
import shap
import re

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Mental Health AI Detection API",
    description="AI-powered mental health condition detection using SVC + SHAP explainability",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://mental-health-ai-bice.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBALS
# =========================

models = {}
explainers = {}

# =========================
# LABELS
# =========================

LABELS = [
    'addiction',
    'adhd',
    'anxiety',
    'autism',
    'bipolar',
    'bpd',
    'depression',
    'ocd',
    'psychosis',
    'ptsd',
    'suicide'
]

ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# =========================
# MODEL INFO
# =========================

MODEL_INFO = {
    "svc": {
        "name": "Support Vector Classifier",
        "description": "Linear SVC using TF-IDF word + character n-grams",
        "accuracy": 0.78,
        "type": "advanced"
    },
    "ensemble": {
        "name": "Ensemble",
        "description": "Production ensemble using SVC pipeline",
        "accuracy": 0.78,
        "type": "ensemble"
    }
}

# =========================
# CATEGORY DESCRIPTIONS
# =========================

CATEGORY_DESCRIPTIONS = {
    "addiction": "Substance abuse and addictive behaviors",
    "adhd": "Attention Deficit Hyperactivity Disorder",
    "anxiety": "Anxiety disorders and panic symptoms",
    "autism": "Autism Spectrum Disorder",
    "bipolar": "Bipolar disorder and mood instability",
    "bpd": "Borderline Personality Disorder",
    "depression": "Major depressive disorder",
    "ocd": "Obsessive Compulsive Disorder",
    "psychosis": "Psychotic disorders",
    "ptsd": "Post-Traumatic Stress Disorder",
    "suicide": "Suicidal ideation and self-harm"
}

# =========================
# REQUEST MODELS
# =========================

class TextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

    model: Literal["svc", "ensemble"] = Field(
        default="ensemble"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=11
    )


class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., max_items=50)

    model: Literal["svc", "ensemble"] = Field(
        default="ensemble"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=11
    )


class ExplainInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

    model: Literal["svc"] = Field(
        default="svc"
    )

    max_display: int = Field(
        default=20,
        ge=5,
        le=100
    )

# =========================
# RESPONSE MODELS
# =========================

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


# =========================
# STARTUP
# =========================

@app.on_event("startup")
async def load_models():
    global models, explainers

    try:
        current_file = Path(__file__).resolve()
        backend_dir = current_file.parent
        MODEL_DIR = backend_dir / "src" / "models"

        print("\n" + "=" * 60)
        print("🚀 Starting Mental Health AI Backend")
        print("=" * 60)

        required_files = [
            "svc_model.pkl",
            "tfidf_word.pkl",
            "tfidf_char.pkl"
        ]

        for filename in required_files:
            filepath = MODEL_DIR / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Missing required file: {filename}"
                )

        print("📥 Loading models...")

        models["svc"] = joblib.load(
            MODEL_DIR / "svc_model.pkl"
        )

        models["tfidf_word"] = joblib.load(
            MODEL_DIR / "tfidf_word.pkl"
        )

        models["tfidf_char"] = joblib.load(
            MODEL_DIR / "tfidf_char.pkl"
        )

        print(
            f"✅ Word vocab: {len(models['tfidf_word'].vocabulary_)}"
        )

        print(
            f"✅ Char vocab: {len(models['tfidf_char'].vocabulary_)}"
        )

        # =========================
        # SHAP EXPLAINER
        # =========================

        def predict_fn(texts):
            X_word = models["tfidf_word"].transform(texts)

            X_char = models["tfidf_char"].transform(texts)

            X_combined = hstack([X_word, X_char])

            return models["svc"].predict_proba(X_combined)

        masker = shap.maskers.Text(tokenizer=r"\W+")

        explainers["svc"] = shap.Explainer(
            predict_fn,
            masker,
            output_names=LABELS
        )

        print("✅ SHAP explainer loaded")
        print("✅ Backend ready")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        traceback.print_exc()

# =========================
# FEATURE PREP
# =========================

def prepare_features(texts: List[str]):

    try:
        X_word = models["tfidf_word"].transform(texts)

        X_char = models["tfidf_char"].transform(texts)

        X_combined = hstack([X_word, X_char])

        return X_combined

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature preparation failed: {str(e)}"
        )

# =========================
# PREDICTION
# =========================

def predict_with_model(
    X,
    texts: List[str],
    model_name: str,
    top_k: int = 3
):

    try:
        probs = models["svc"].predict_proba(X)

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
                "text": text[:100] + "..." if len(text) > 100 else text,
                "model_used": model_name,
                "predictions": predictions,
                "top_prediction": predictions[0]["category"],
                "confidence": predictions[0]["confidence"],
                "model_info": MODEL_INFO[model_name]
            })

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# =========================
# ROOT
# =========================

@app.get("/")
async def root():

    return {
        "status": "active",
        "service": "Mental Health AI Detection API",
        "models_loaded": len(models) > 0,
        "available_models": list(MODEL_INFO.keys()),
        "categories": LABELS
    }

# =========================
# HEALTH
# =========================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "explainers_loaded": list(explainers.keys())
    }

# =========================
# MODELS
# =========================

@app.get("/models")
async def get_models():

    return {
        "models": MODEL_INFO
    }

# =========================
# CATEGORIES
# =========================

@app.get("/categories")
async def get_categories():

    return {
        "categories": CATEGORY_DESCRIPTIONS
    }

# =========================
# SINGLE PREDICTION
# =========================

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput):

    try:
        X = prepare_features([input_data.text])

        results = predict_with_model(
            X,
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
            detail=str(e)
        )

# =========================
# BATCH PREDICTION
# =========================

@app.post("/predict/batch")
async def predict_batch(input_data: BatchTextInput):

    try:
        X = prepare_features(input_data.texts)

        results = predict_with_model(
            X,
            input_data.texts,
            input_data.model,
            input_data.top_k
        )

        return {
            "predictions": results,
            "count": len(results)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================
# MODEL COMPARISON
# =========================

@app.post("/predict/compare", response_model=ModelComparisonResponse)
async def compare_models(input_data: TextInput):

    try:
        X = prepare_features([input_data.text])

        results = {}

        for model_name in MODEL_INFO.keys():

            prediction = predict_with_model(
                X,
                [input_data.text],
                model_name,
                3
            )[0]

            results[model_name] = {
                "top_prediction": prediction["top_prediction"],
                "confidence": prediction["confidence"],
                "top_predictions": prediction["predictions"]
            }

        return {
            "text": input_data.text,
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================
# EXPLAIN
# =========================

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(input_data: ExplainInput):

    try:
        text = input_data.text

        explainer = explainers["svc"]

        X = prepare_features([text])

        prediction = predict_with_model(
            X,
            [text],
            "svc",
            1
        )[0]

        predicted_category = prediction["top_prediction"]

        confidence = prediction["confidence"]

        shap_values = explainer([text])

        predicted_idx = LABEL2ID[predicted_category]

        tokens = re.split(r"\W+", text.lower())

        tokens = [t for t in tokens if t]

        token_shap_values = shap_values[0][:, predicted_idx].values

        base_value = float(
            shap_values[0][:, predicted_idx].base_values
        )

        feature_importances = []

        for i, (token, importance) in enumerate(
            zip(tokens, token_shap_values)
        ):

            if i >= input_data.max_display:
                break

            feature_importances.append({
                "token": token,
                "importance": float(importance),
                "position": i
            })

        feature_importances.sort(
            key=lambda x: abs(x["importance"]),
            reverse=True
        )

        return {
            "text": text,
            "model_used": "svc",
            "predicted_category": predicted_category,
            "confidence": confidence,
            "base_value": base_value,
            "feature_importances": feature_importances,
            "explanation_type": "SHAP"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )

# =========================
# ERROR HANDLERS
# =========================

@app.exception_handler(404)
async def not_found_handler(request, exc):

    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error"
        }
    )

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )