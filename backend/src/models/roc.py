import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Configuration
DATA_PATH = '../../../data/processed/cleaned_combined_data_utf8.csv'
MODEL_DIR = '../../../backend/src/models'
SEED = 42
TEST_SIZE = 0.2

categories = ["depression", "anxiety", "ocd", "adhd", "bipolar", 
              "addiction", "autism", "bpd", "psychosis", "ptsd", "suicide"]

print("="*60)
print("ADVANCED ROC-AUC GENERATION WITH FEATURE-RICH MODELS")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PATH).dropna(subset=['combined_text', 'category']).copy()
print(f"✓ Loaded {len(df)} samples")

X = df['combined_text']
y = df['category']

# Split data
print(f"\nSplitting data (test_size={TEST_SIZE}, random_state={SEED})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Build advanced TF-IDF features (like Script 2)
print("\n" + "="*60)
print("BUILDING ADVANCED TF-IDF FEATURES")
print("="*60)

# Word-level TF-IDF
print("\n1. Word-level TF-IDF (up to 120k features)...")
word_vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=120000,
    lowercase=True,
    strip_accents="unicode",
    min_df=2,
    max_df=0.9,
    stop_words="english"
)
Xtr_w = word_vec.fit_transform(X_train)
Xte_w = word_vec.transform(X_test)
print(f"   Shape: {Xtr_w.shape}")

# Character-level TF-IDF
print("\n2. Character-level TF-IDF (up to 180k features)...")
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=180000,
    min_df=2
)
Xtr_c = char_vec.fit_transform(X_train)
Xte_c = char_vec.transform(X_test)
print(f"   Shape: {Xtr_c.shape}")

# Combine features
print("\n3. Combining features...")
X_train_vec = hstack([Xtr_w, Xtr_c]).tocsr()
X_test_vec = hstack([Xte_w, Xte_c]).tocsr()
print(f"   Combined shape: {X_train_vec.shape}")

# Save vectorizers
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(word_vec, os.path.join(MODEL_DIR, 'tfidf_word.pkl'))
joblib.dump(char_vec, os.path.join(MODEL_DIR, 'tfidf_char.pkl'))
print("✓ Saved vectorizers")

# Train models
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

print("\n1. Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
lr.fit(X_train_vec, y_train)
joblib.dump(lr, os.path.join(MODEL_DIR, 'baseline_logistic_regression.pkl'))
print("✓ LR trained and saved")

print("\n2. Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, 
                            class_weight='balanced', max_depth=20)
rf.fit(X_train_vec, y_train)
joblib.dump(rf, os.path.join(MODEL_DIR, 'baseline_random_forest.pkl'))
print("✓ RF trained and saved")

print("\n3. Training Calibrated SVC...")
svc_base = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000)
svc = CalibratedClassifierCV(svc_base, method="sigmoid", cv=3)
svc.fit(X_train_vec, y_train)
joblib.dump(svc, os.path.join(MODEL_DIR, 'svc_model.pkl'))
print("✓ SVC trained and saved")

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=categories)

# Calculate per-class AUC
print("\n" + "="*60)
print("CALCULATING ROC-AUC SCORES")
print("="*60)

models_dict = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'SVC': svc
}

results = {cat: {} for cat in categories}
micro_avg = {}
macro_avg = {}

for name, model in models_dict.items():
    print(f"\nProcessing {name}...")
    y_score = model.predict_proba(X_test_vec)
    
    # Per-class AUC
    class_aucs = []
    for i, cat in enumerate(categories):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        class_auc = auc(fpr, tpr)
        results[cat][name] = class_auc
        class_aucs.append(class_auc)
    
    # Micro-average AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    micro_avg[name] = auc(fpr_micro, tpr_micro)
    
    # Macro-average AUC
    macro_avg[name] = np.mean(class_aucs)
    
    print(f"  Micro-Average AUC: {micro_avg[name]:.4f}")
    print(f"  Macro-Average AUC: {macro_avg[name]:.4f}")

# Create results table
df_results = pd.DataFrame(results).T
df_results.columns = ['LR', 'RF', 'SVC']
df_results.loc['Micro-Average'] = [micro_avg['Logistic Regression'], 
                                    micro_avg['Random Forest'], 
                                    micro_avg['SVC']]
df_results = df_results.sort_values('SVC', ascending=False)

# Print table
print("\n" + "="*60)
print("PER-CLASS ROC-AUC SCORES")
print("="*60)
print(df_results.to_string(float_format='%.4f'))

# Save results
df_results.to_csv('per_class_auc_advanced.csv', float_format='%.4f')
print("\n✓ Saved to per_class_auc_advanced.csv")

# Generate LaTeX table (clean version)
print("\n" + "="*60)
print("LATEX TABLE")
print("="*60)

# Create clean table with proper formatting
latex_output = "\\begin{table}[htbp]\n\\centering\n"
latex_output += "\\caption{Per-Class ROC-AUC Scores Across Different Mental Health Categories}\n"
latex_output += "\\label{tab:per_class_auc}\n"
latex_output += "\\begin{tabular}{lrrr}\n\\toprule\n"
latex_output += "\\textbf{Category} & \\textbf{LR} & \\textbf{RF} & \\textbf{SVC} \\\\\n\\midrule\n"

for idx, row in df_results.iterrows():
    if idx == 'Micro-Average':
        latex_output += "\\midrule\n"
    latex_output += f"{idx} & {row['LR']:.3f} & {row['RF']:.3f} & {row['SVC']:.3f} \\\\\n"

latex_output += "\\bottomrule\n\\end{tabular}\n\\end{table}"

print(latex_output)

# Plot ROC curve (micro-average)
print("\n" + "="*60)
print("GENERATING ROC CURVE")
print("="*60)

plt.figure(figsize=(10, 8))

for name, model in models_dict.items():
    y_score = model.predict_proba(X_test_vec)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-AUC Curve Comparison (Advanced Features)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_auc_advanced.png', dpi=300)
print("✓ Saved ROC curve to roc_auc_advanced.png")

# Calculate accuracy, precision, recall, F1
print("\n" + "="*60)
print("ACCURACY, PRECISION, RECALL, F1-SCORE")
print("="*60)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

for name, model in models_dict.items():
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\n{name}:")
    print(f"  Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro-Avg Precision:  {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"  Macro-Avg Recall:     {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"  Macro-Avg F1:         {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Weighted-Avg Prec:    {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    print(f"  Weighted-Avg Recall:  {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    print(f"  Weighted-Avg F1:      {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"\nFiles saved:")
print(f"  - Models: {MODEL_DIR}/")
print(f"  - ROC curve: roc_auc_advanced.png")
print(f"  - AUC table: per_class_auc_advanced.csv")

plt.show()