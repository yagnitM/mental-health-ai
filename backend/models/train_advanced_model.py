import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import sys
import json
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack
from packaging import version
import sklearn

try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except ImportError:
    USE_SBERT = False
    print("⚠ sentence-transformers not installed. SBERT model will be skipped.")
    print("  Install with: pip install sentence-transformers")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "../../data/processed/cleaned_combined_data_utf8.csv"
CACHE_DIR = "../../data/processed"
MODEL_DIR = "../src/models"
SEED = 42
TEST_SIZE = 0.2
MAX_WORD_FEATURES = 120_000
MAX_CHAR_FEATURES = 180_000
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH = 128

# ============================================================
# Helper Functions
# ============================================================

def load_df():
    """Load and clean the dataset."""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["combined_text", "category"]).copy()
    print(f"✓ Loaded {len(df)} samples")
    return df


def get_label_maps(categories):
    """Create label encoding mappings."""
    labels = sorted(list(categories))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def build_tfidf(X_train_text, X_test_text):
    """Build word-level and character-level TF-IDF features."""
    print("\nBuilding TF-IDF features...")
    
    # Word-level TF-IDF
    print("  - Word-level TF-IDF...")
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=MAX_WORD_FEATURES,
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.9,
        stop_words="english"
    )
    Xtr_w = word_vec.fit_transform(X_train_text)
    Xte_w = word_vec.transform(X_test_text)
    print(f"    Shape: {Xtr_w.shape}")
    
    # Character-level TF-IDF
    print("  - Character-level TF-IDF...")
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=MAX_CHAR_FEATURES,
        min_df=2
    )
    Xtr_c = char_vec.fit_transform(X_train_text)
    Xte_c = char_vec.transform(X_test_text)
    print(f"    Shape: {Xtr_c.shape}")
    
    # Combine features
    print("  - Combining features...")
    Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
    Xte = hstack([Xte_w, Xte_c]).tocsr()
    print(f"    Combined shape: {Xtr.shape}")
    
    return (Xtr, Xte), (word_vec, char_vec)


def train_calibrated_svc(Xtr, ytr):
    """Train a calibrated LinearSVC model."""
    print("\nTraining Calibrated LinearSVC...")
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000)
    
    # Handle different sklearn versions
    if version.parse(sklearn.__version__) >= version.parse("1.4"):
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    else:
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    
    clf.fit(Xtr, ytr)
    print("✓ SVC training complete")
    return clf


def sbert_encode(model, texts, batch=128):
    """Encode texts using SBERT."""
    embs = model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embs.astype("float32")


def get_cached_embeddings(model_name, split_name, texts, batch):
    """Load cached embeddings or generate and cache them."""
    cache_path = os.path.join(CACHE_DIR, f"{split_name}_{model_name.replace('/', '_')}.npy")
    
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from: {cache_path}")
        return np.load(cache_path)
    
    print(f"  Generating embeddings...")
    model = SentenceTransformer(model_name)
    embs = sbert_encode(model, texts, batch=batch)
    
    print(f"  Saving embeddings to: {cache_path}")
    np.save(cache_path, embs)
    return embs


def train_lr_on_embeddings(Xtr, ytr):
    """Train Logistic Regression on SBERT embeddings."""
    print("\nTraining Logistic Regression on SBERT embeddings...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="saga",
        random_state=SEED
    )
    lr.fit(Xtr, ytr)
    print("✓ LR training complete")
    return lr


def evaluate_model(model, X, y_true, model_name):
    """Evaluate a model and print results."""
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return {"accuracy": acc, "f1": f1, "predictions": y_pred}


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    start_time = time.time()
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("="*60)
    print("ADVANCED MODEL TRAINING PIPELINE")
    print("="*60)
    
    # --------------------------------------------------------
    # 1. Load and prepare data
    # --------------------------------------------------------
    df = load_df()
    
    # Get label mappings
    labels, label2id, id2label = get_label_maps(df["category"].unique())
    df["label"] = df["category"].map(label2id)
    print(f"\nLabel mapping: {label2id}")
    print(f"Class distribution:\n{df['category'].value_counts()}")
    
    # Save label mappings
    label_map_path = os.path.join(MODEL_DIR, "label_mapping.json")
    with open(label_map_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"\n✓ Saved label mappings to: {label_map_path}")
    
    # --------------------------------------------------------
    # 2. Train/Test Split
    # --------------------------------------------------------
    print(f"\nSplitting data (test_size={TEST_SIZE}, random_state={SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=SEED
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set:  {len(X_test)} samples")
    
    # Save test set for evaluation
    test_df = pd.DataFrame({
        'combined_text': X_test.values,
        'label': y_test.values,
        'category': y_test.map(id2label).values
    })
    test_set_path = os.path.join(CACHE_DIR, "test_set.csv")
    test_df.to_csv(test_set_path, index=False)
    print(f"✓ Saved test set to: {test_set_path}")
    
    # --------------------------------------------------------
    # 3. Build TF-IDF Features
    # --------------------------------------------------------
    (Xtr_tfidf, Xte_tfidf), (word_vec, char_vec) = build_tfidf(X_train, X_test)
    
    # Save vectorizers
    joblib.dump(word_vec, os.path.join(MODEL_DIR, "tfidf_word.pkl"))
    joblib.dump(char_vec, os.path.join(MODEL_DIR, "tfidf_char.pkl"))
    print("✓ Saved TF-IDF vectorizers")
    
    # --------------------------------------------------------
    # 4. Train SVC Model (TF-IDF)
    # --------------------------------------------------------
    svc_model = train_calibrated_svc(Xtr_tfidf, y_train.values)
    
    # Evaluate on test set
    svc_results = evaluate_model(svc_model, Xte_tfidf, y_test.values, "SVC (TF-IDF)")
    
    # Save model
    svc_model_path = os.path.join(MODEL_DIR, "svc_model.pkl")
    joblib.dump(svc_model, svc_model_path)
    print(f"✓ Saved SVC model to: {svc_model_path}")
    
    # --------------------------------------------------------
    # 5. Train LR Model on SBERT Embeddings (if available)
    # --------------------------------------------------------
    if USE_SBERT:
        print("\n" + "="*60)
        print("SBERT-BASED MODEL")
        print("="*60)
        
        # Generate/load embeddings
        print("\nGenerating SBERT embeddings for training set...")
        tr_emb = get_cached_embeddings(
            SBERT_MODEL,
            "train_full",
            list(X_train),
            SBERT_BATCH
        )
        print(f"✓ Training embeddings shape: {tr_emb.shape}")
        
        print("\nGenerating SBERT embeddings for test set...")
        te_emb = get_cached_embeddings(
            SBERT_MODEL,
            "test_full",
            list(X_test),
            SBERT_BATCH
        )
        print(f"✓ Test embeddings shape: {te_emb.shape}")
        
        # Train model
        lr_sbert_model = train_lr_on_embeddings(tr_emb, y_train.values)
        
        # Evaluate on test set
        lr_results = evaluate_model(lr_sbert_model, te_emb, y_test.values, "LR (SBERT)")
        
        # Save model
        lr_model_path = os.path.join(MODEL_DIR, "lr_sbert.pkl")
        joblib.dump(lr_sbert_model, lr_model_path)
        print(f"✓ Saved LR model to: {lr_model_path}")
    else:
        lr_results = None
    
    # --------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"\nTest Set Performance:")
    print(f"  SVC (TF-IDF): Accuracy={svc_results['accuracy']:.4f}, F1={svc_results['f1']:.4f}")
    if lr_results:
        print(f"  LR (SBERT):   Accuracy={lr_results['accuracy']:.4f}, F1={lr_results['f1']:.4f}")
    
    print(f"\nSaved artifacts:")
    print(f"  - Models: {MODEL_DIR}")
    print(f"  - Test set: {test_set_path}")
    print(f"  - Label mappings: {label_map_path}")
    print(f"  - Embeddings cache: {CACHE_DIR}")
    
    print("\n✅ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()