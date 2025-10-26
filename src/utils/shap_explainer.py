import os
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")


class MentalHealthExplainer:
    def __init__(self, model_dir=None):
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(current_dir, "..", "models")
        else:
            self.model_dir = model_dir
        self.LABELS = [
            'addiction', 'adhd', 'anxiety', 'autism', 'bipolar',
            'bpd', 'depression', 'ocd', 'psychosis', 'ptsd', 'suicide'
        ]
        self.ID2LABEL = {i: label for i, label in enumerate(self.LABELS)}
        self.models = {}
        self.explainers = {}
        self._load_models()

    def _load_models(self):
        print("Loading models for SHAP analysis...")
        self.models['svc'] = joblib.load(os.path.join(self.model_dir, "svc_model.pkl"))
        self.models['tfidf_word'] = joblib.load(os.path.join(self.model_dir, "tfidf_word.pkl"))
        self.models['tfidf_char'] = joblib.load(os.path.join(self.model_dir, "tfidf_char.pkl"))
        self.models['lr_sbert'] = joblib.load(os.path.join(self.model_dir, "lr_sbert.pkl"))
        self.models['baseline_lr'] = joblib.load(os.path.join(self.model_dir, "baseline_logistic_regression.pkl"))
        self.models['baseline_rf'] = joblib.load(os.path.join(self.model_dir, "baseline_random_forest.pkl"))
        self.models['baseline_vectorizer'] = joblib.load(os.path.join(self.model_dir, "tfidf_vectorizer.pkl"))
        self.models['sbert_model'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Models loaded successfully!")

    def _prepare_features(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X_word = self.models['tfidf_word'].transform(texts)
        X_char = self.models['tfidf_char'].transform(texts)
        X_advanced = hstack([X_word, X_char])
        embeddings = self.models['sbert_model'].encode(texts, batch_size=32, convert_to_numpy=True)
        X_baseline = self.models['baseline_vectorizer'].transform(texts)
        return {'advanced': X_advanced, 'sbert': embeddings, 'baseline': X_baseline}

    def explain_baseline_lr(self, text, num_features=20):
        print("\n📊 Explaining Baseline Logistic Regression...")
        features = self._prepare_features(text)
        X = features['baseline']
        background = csr_matrix((1, X.shape[1]), dtype=X.dtype)
        explainer = shap.LinearExplainer(self.models['baseline_lr'], background)
        shap_values = explainer(X)
        feature_names = self.models['baseline_vectorizer'].get_feature_names_out()
        pred_probs = self.models['baseline_lr'].predict_proba(X)[0]
        pred_class = np.argmax(pred_probs)
        pred_label = self.ID2LABEL[pred_class]
        print(f"Predicted: {pred_label.upper()} ({pred_probs[pred_class]:.2%})")
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_names,
            'data_row': X[0].toarray().ravel(),
            'prediction': pred_label,
            'confidence': pred_probs[pred_class],
            'all_probs': pred_probs
        }

    def explain_baseline_rf(self, text, background_data=None, num_features=20):
        print("\n🌳 Explaining Baseline Random Forest...")
        features = self._prepare_features(text)
        X_sparse = features['baseline']
        X = X_sparse.toarray()
        explainer = shap.TreeExplainer(self.models['baseline_rf'])
        shap_values = explainer(X)
        feature_names = self.models['baseline_vectorizer'].get_feature_names_out()
        pred_probs = self.models['baseline_rf'].predict_proba(X)[0]
        pred_class = np.argmax(pred_probs)
        pred_label = self.ID2LABEL[pred_class]
        print(f"Predicted: {pred_label.upper()} ({pred_probs[pred_class]:.2%})")
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_names,
            'data_row': X[0],
            'prediction': pred_label,
            'confidence': pred_probs[pred_class],
            'all_probs': pred_probs
        }

    def explain_svc(self, text, background_data=None, n_samples=100):
        print("\n⚙️ Explaining SVC (this may take a moment)...")
        features = self._prepare_features(text)
        X = features['advanced']
        if background_data is None:
            print("⚠️ Warning: Using current sample as background. Pass background_data for better results.")
            background = X
        else:
            background_features = self._prepare_features(background_data)
            background = background_features['advanced']
        def predict_fn(Xi): return self.models['svc'].predict_proba(Xi)
        explainer = shap.KernelExplainer(predict_fn, background, link="identity")
        shap_values = explainer.shap_values(X, nsamples=n_samples)
        pred_probs = self.models['svc'].predict_proba(X)[0]
        pred_class = np.argmax(pred_probs)
        pred_label = self.ID2LABEL[pred_class]
        print(f"Predicted: {pred_label.upper()} ({pred_probs[pred_class]:.2%})")
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'prediction': pred_label,
            'confidence': pred_probs[pred_class],
            'all_probs': pred_probs
        }

    def explain_sbert_lr(self, text, num_features=20):
        print("\n🤖 Explaining SBERT-LR...")
        features = self._prepare_features(text)
        X = features['sbert']
        background = np.zeros((1, X.shape[1]))
        explainer = shap.LinearExplainer(self.models['lr_sbert'], background)
        shap_values = explainer(X)
        pred_probs = self.models['lr_sbert'].predict_proba(X)[0]
        pred_class = np.argmax(pred_probs)
        pred_label = self.ID2LABEL[pred_class]
        print(f"Predicted: {pred_label.upper()} ({pred_probs[pred_class]:.2%})")
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': [f"embed_{i}" for i in range(X.shape[1])],
            'data_row': X[0],
            'prediction': pred_label,
            'confidence': pred_probs[pred_class],
            'all_probs': pred_probs
        }

    def _build_single_row_explanation(self, explanation_result, target_class=None):
        shap_values_obj = explanation_result['shap_values']
        feature_names = explanation_result['feature_names']
        data_row = explanation_result['data_row']
        if target_class is None:
            target_class = int(np.argmax(explanation_result['all_probs']))
        if isinstance(shap_values_obj, shap.Explanation):
            vals = shap_values_obj.values
            base = shap_values_obj.base_values
            if vals.ndim == 3:
                values_1d = vals[0, :, target_class]
                if isinstance(base, np.ndarray):
                    base_val = base[0, target_class] if base.ndim == 2 else base[0]
                else:
                    base_val = float(base)
            elif vals.ndim == 2:
                values_1d = vals[0, :]
                if isinstance(base, np.ndarray):
                    base_val = base[0] if base.ndim == 1 else base[0, 0]
                else:
                    base_val = float(base)
            else:
                values_1d = vals
                base_val = float(base) if not isinstance(base, np.ndarray) else np.ravel(base)[0]
            return shap.Explanation(values=values_1d, base_values=base_val, data=np.asarray(data_row), feature_names=list(feature_names))
        if isinstance(shap_values_obj, list):
            sv = shap_values_obj[target_class]
            if isinstance(sv, shap.Explanation):
                values_1d = sv.values[0, :] if sv.values.ndim == 2 else np.ravel(sv.values)
                base = sv.base_values
                base_val = base[0] if isinstance(base, np.ndarray) else float(base)
            else:
                arr = np.asarray(sv)
                values_1d = arr[0, :] if arr.ndim == 2 else np.ravel(arr)
                base_val = 0.0
            return shap.Explanation(values=values_1d, base_values=base_val, data=np.asarray(data_row), feature_names=list(feature_names))
        arr = np.asarray(shap_values_obj)
        if arr.ndim == 3:
            values_1d = arr[0, :, target_class]
        elif arr.ndim == 2:
            values_1d = arr[0, :]
        else:
            values_1d = np.ravel(arr)
        base_val = 0.0
        return shap.Explanation(values=values_1d, base_values=base_val, data=np.asarray(data_row), feature_names=list(feature_names))

    def plot_bar(self, explanation_result, target_class=None, max_display=20):
        ex_row = self._build_single_row_explanation(explanation_result, target_class)
        plt.close('all')
        try:
            shap.plots.bar(ex_row, max_display=max_display, show=False)
        except TypeError:
            shap.plots.bar(ex_row, max_display=max_display)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Feature Importance - {self.ID2LABEL[np.argmax(explanation_result['all_probs'])].upper()}")
        fig.canvas.draw()
        fig.tight_layout()
        return fig

    def plot_waterfall(self, explanation_result, target_class=None, max_display=20):
        ex_row = self._build_single_row_explanation(explanation_result, target_class)
        plt.close('all')
        try:
            shap.plots.waterfall(ex_row, max_display=max_display, show=False)
        except TypeError:
            shap.plots.waterfall(ex_row, max_display=max_display)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Waterfall Plot - {self.ID2LABEL[np.argmax(explanation_result['all_probs'])].upper()}")
        fig.canvas.draw()
        fig.tight_layout()
        return fig

    def plot_force(self, explanation_result, target_class=None):
        ex_row = self._build_single_row_explanation(explanation_result, target_class)
        return shap.plots.force(ex_row, matplotlib=True)

    def explain_all_models(self, text, save_plots=False, output_dir="../../reports/shap_explanations"):
        print(f"\n{'='*60}")
        print(f"Analyzing text: '{text[:100]}...'")
        print(f"{'='*60}")
        explanations = {}
        explanations['baseline_lr'] = self.explain_baseline_lr(text)
        explanations['baseline_rf'] = self.explain_baseline_rf(text)
        explanations['sbert_lr'] = self.explain_sbert_lr(text)
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            for model_name, result in explanations.items():
                try:
                    fig = self.plot_bar(result, max_display=15)
                    fig.savefig(f"{output_dir}/{model_name}_bar.jpg", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    fig = self.plot_waterfall(result, max_display=15)
                    fig.savefig(f"{output_dir}/{model_name}_waterfall.jpg", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Could not save plot for {model_name}: {e}")
            print(f"\n✅ Plots saved to {output_dir}/")
        return explanations

    def get_top_features_text(self, explanation_result, num_features=10):
        ex_row = self._build_single_row_explanation(explanation_result)
        feature_names = explanation_result.get('feature_names', None)
        if feature_names is None:
            return "Feature names not available for this model"
        values = ex_row.values
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[-num_features:][::-1]
        pred_class = np.argmax(explanation_result['all_probs'])
        result_text = f"\nTop {num_features} features for {self.ID2LABEL[pred_class].upper()}:\n"
        result_text += "=" * 50 + "\n"
        for rank, idx in enumerate(top_indices, 1):
            feature = feature_names[idx]
            shap_val = values[idx]
            direction = "→" if shap_val > 0 else "←"
            result_text += f"{rank:2d}. {feature:30s} {direction} {abs(shap_val):>8.4f}\n"
        return result_text

    def english_explanation(self, explanation_result, top_k=5):
        ex_row = self._build_single_row_explanation(explanation_result)
        names = list(explanation_result['feature_names'])
        vals = ex_row.values
        idx = np.argsort(np.abs(vals))[::-1][:top_k]
        pos = [names[i] for i in idx if vals[i] > 0]
        neg = [names[i] for i in idx if vals[i] < 0]
        pred_idx = int(np.argmax(explanation_result['all_probs']))
        label = self.ID2LABEL[pred_idx]
        conf = float(explanation_result['confidence'])
        pos_str = ", ".join(f"“{w}”" for w in pos) if pos else "no strong positive signals"
        neg_str = ", ".join(f"“{w}”" for w in neg) if neg else "no strong negative signals"
        return f"The model predicted {label.upper()} with confidence {conf:.1%}. This was mainly increased by {pos_str}, and decreased by {neg_str}."


def main():
    explainer = MentalHealthExplainer()
    text = "I can't stop checking if I locked the door and washing my hands repeatedly"
    explanations = explainer.explain_all_models(text, save_plots=True)
    print(explainer.get_top_features_text(explanations['baseline_lr'], num_features=10))
    print(explainer.get_top_features_text(explanations['sbert_lr'], num_features=10))
    print(explainer.english_explanation(explanations['baseline_lr'], top_k=5))
    print(explainer.english_explanation(explanations['sbert_lr'], top_k=5))


if __name__ == "__main__":
    main()
