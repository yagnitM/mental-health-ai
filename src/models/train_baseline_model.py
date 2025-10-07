import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from datetime import datetime
import json

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {df['category'].value_counts()}")
    return df

def preprocess_data(df, text_column='combined_text', target_column='category'):
    df = df.dropna(subset=[text_column, target_column])
    X = df[text_column]
    y = df[target_column]
    print(f"Final dataset size: {len(X)} samples")
    print(f"Class distribution:\n{y.value_counts()}")
    return X, y

def create_tfidf_features(X_train, X_test, max_features=10000, ngram_range=(1, 2)):
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        max_df=0.95,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_models(X_train, y_train, X_test, y_test):
    models = {}
    results = {}
    
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'predictions': lr_pred,
        'report': classification_report(y_test, lr_pred, output_dict=True)
    }
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=20
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'predictions': rf_pred,
        'report': classification_report(y_test, rf_pred, output_dict=True)
    }
    
    return models, results

def save_models(models, vectorizer, results, model_dir='../src/models'):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    joblib.dump(models['logistic_regression'], f'{model_dir}/baseline_logistic_regression.pkl')
    joblib.dump(models['random_forest'], f'{model_dir}/baseline_random_forest.pkl')
    joblib.dump(vectorizer, f'{model_dir}/tfidf_vectorizer.pkl')
    
    results_summary = {
        'timestamp': timestamp,
        'logistic_regression_accuracy': results['logistic_regression']['accuracy'],
        'random_forest_accuracy': results['random_forest']['accuracy'],
        'models_saved': ['baseline_logistic_regression.pkl', 'baseline_random_forest.pkl'],
        'vectorizer_saved': 'tfidf_vectorizer.pkl'
    }
    
    with open(f'{model_dir}/baseline_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nModels and results saved in {model_dir}/")
    print(f"Logistic Regression Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    print(f"Random Forest Accuracy: {results['random_forest']['accuracy']:.4f}")

def main():
    print("Mental Health Baseline Model Training\n")
    data_path = '../../data/processed/cleaned_combined_data_utf8.csv'
    df = load_data(data_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)
    models, results = train_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    save_models(models, vectorizer, results, model_dir='../src/models')
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()