import os, sys, json, time, joblib, warnings
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

DATA_PATH = "../../data/processed/cleaned_combined_data_utf8.csv"
CACHE_DIR = "../../data/processed"
MODEL_DIR = "../src/models"
SEED = 42
TEST_SIZE = 0.2
MAX_WORD_FEATURES = 120_000
MAX_CHAR_FEATURES = 180_000
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH = 128

def load_df():
    df = pd.read_csv(DATA_PATH).dropna(subset=["combined_text", "category"]).copy()
    return df

def get_label_maps(categories):
    labels = sorted(list(categories))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return labels, label2id, id2label

def build_tfidf(X_train_text, X_val_text):
    word_vec = TfidfVectorizer(
        ngram_range=(1,2), max_features=MAX_WORD_FEATURES,
        lowercase=True, strip_accents="unicode",
        min_df=2, max_df=0.9, stop_words="english"
    )
    Xtr_w = word_vec.fit_transform(X_train_text)
    Xval_w = word_vec.transform(X_val_text)

    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5),
        max_features=MAX_CHAR_FEATURES, min_df=2
    )
    Xtr_c = char_vec.fit_transform(X_train_text)
    Xval_c = char_vec.transform(X_val_text)

    Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
    Xval = hstack([Xval_w, Xval_c]).tocsr()
    return (Xtr, Xval), (word_vec, char_vec)

def train_calibrated_svc(Xtr, ytr):
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
    if version.parse(sklearn.__version__) >= version.parse("1.4"):
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    else:
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)
    return clf

def sbert_encode(model, texts, batch=128):
    embs = model.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True)
    return embs.astype("float32")

def get_cached_embeddings(model_name, split_name, texts, batch):
    cache_path = os.path.join(CACHE_DIR, f"{split_name}_{model_name.replace('/','_')}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    embs = sbert_encode(SentenceTransformer(model_name), texts, batch=batch)
    np.save(cache_path, embs)
    return embs

def train_lr_on_embeddings(Xtr, ytr):
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, solver="saga")
    lr.fit(Xtr, ytr)
    return lr

def main():
    start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading data...")
    df = load_df()
    labels, label2id, id2label = get_label_maps(df["category"].unique())
    df["label"] = df["category"].map(label2id)
    print(f"Found labels: {labels}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"], df["label"], test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )

    print("Building TF-IDF features...")
    (Xtr_tfidf, Xte_tfidf), vecs = build_tfidf(X_train, X_test)

    print("Training Calibrated LinearSVC...")
    svc_model = train_calibrated_svc(Xtr_tfidf, y_train.values)
    
    joblib.dump(svc_model, os.path.join(MODEL_DIR, "svc_model.pkl"))
    joblib.dump(vecs[0], os.path.join(MODEL_DIR, "tfidf_word.pkl"))
    joblib.dump(vecs[1], os.path.join(MODEL_DIR, "tfidf_char.pkl"))
    print("SVC model and TF-IDF vectorizers saved.")

    if USE_SBERT:
        print("Encoding SBERT embeddings...")
        tr_emb = get_cached_embeddings(SBERT_MODEL, "train_full", list(X_train), SBERT_BATCH)
        
        print("Training Logistic Regression on SBERT embeddings...")
        lr_sbert_model = train_lr_on_embeddings(tr_emb, y_train.values)
        joblib.dump(lr_sbert_model, os.path.join(MODEL_DIR, "lr_sbert.pkl"))
        print("SBERT-LR model saved.")
    
    elapsed = time.time() - start
    print(f"\n✅ Done. Training complete in {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()