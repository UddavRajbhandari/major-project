"""
Nepali Hate Speech Model Explainability using LIME & SHAP
Standalone script version for CLI or batch execution
"""

import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import joblib

warnings.filterwarnings("ignore")

# ------------------------- Optional libraries -------------------------------
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Run: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

# ------------------------- Local utils import -------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from utils.preprocessing import preprocess_for_transformer, is_devanagari, roman_to_devanagari, normalize_dirghikaran

# ------------------------- Nepali Font --------------------------------------
FONT_PATH = "fonts/Kalimati.ttf"  # Adjust path
def load_nepali_font(path=FONT_PATH):
    if not os.path.exists(path):
        print(f"⚠️ Nepali font not found: {path}")
        return None
    fontManager.addfont(path)
    fp = FontProperties(fname=path)
    print(f"✓ Loaded Nepali font: {fp.get_name()}")
    return fp

NEPALI_FONT = load_nepali_font()
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "explanations"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------- Model Loader -------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(local_model_path=None, hf_model_id=None):
    if local_model_path and os.path.exists(local_model_path):
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        le = joblib.load(os.path.join(local_model_path, "label_encoder.pkl"))
        print("✅ Model loaded locally")
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
        le_path = hf_hub_download(hf_model_id, "label_encoder.pkl")
        le = joblib.load(le_path)
        print("✅ Model loaded from HuggingFace")
    model.to(DEVICE).eval()
    return model, tokenizer, le

# ------------------------- Token Alignment / Reconstruction -----------------
import regex
def build_token_alignment(text):
    original_tokens = text.split()
    model_tokens, display_tokens = [], []
    for tok in original_tokens:
        norm_tok = re.sub(r"[\u200d\u200c]", "", tok)
        if regex.search(r"\p{Devanagari}", norm_tok) or re.search(r"\w+", norm_tok):
            model_tokens.append(norm_tok)
            display_tokens.append(tok)
    return model_tokens, display_tokens

def apply_nepali_font(ax, nepali_font=NEPALI_FONT, texts=None, is_tick_labels=True):
    if nepali_font is None:
        return
    if is_tick_labels or texts is None:
        for txt in ax.get_yticklabels():
            label_text = txt.get_text()
            if regex.search(r'\p{Devanagari}', label_text):
                txt.set_fontproperties(nepali_font)
    else:
        for txt in texts:
            if regex.search(r'\p{Devanagari}', txt.get_text()):
                txt.set_fontproperties(nepali_font)

# ------------------------- Model Explainer Wrapper --------------------------
class XLMRobertaExplainer:
    def __init__(self, model, tokenizer, label_encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = label_encoder.classes_.tolist()
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [preprocess_for_transformer(t) for t in texts]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        return probs.cpu().numpy()
    
    def predict_with_analysis(self, text):
        probs = self.predict_proba(text)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_label": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {label: float(prob) for label, prob in zip(self.class_names, probs)}
        }

# ------------------------- LIME Explainer ----------------------------------
class LIMEExplainer:
    def __init__(self, model_explainer, nepali_font=NEPALI_FONT):
        self.model_explainer = model_explainer
        self.nepali_font = nepali_font
        self.explainer = LimeTextExplainer(class_names=model_explainer.class_names, random_state=42)
    
    def explain_and_visualize(self, text):
        exp = self.explainer.explain_instance(text, self.model_explainer.predict_proba, num_samples=200)
        token_weights = dict(exp.as_list())
        model_tokens, display_tokens = build_token_alignment(text)
        word_scores = [(disp, sum(val for tok, val in token_weights.items() if tok in mod))
                       for mod, disp in zip(model_tokens, display_tokens)]
        if not word_scores:
            print("⚠️ No explainable words found")
            return
        features, weights = zip(*word_scores)
        y_pos = range(len(features))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(y_pos, weights)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution")
        apply_nepali_font(ax, self.nepali_font)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"lime_explanation_{abs(hash(text)) % 10**8}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✅ LIME image saved to: {out_path}")

# ------------------------- SHAP Explainer ----------------------------------
class SHAPExplainer:
    def __init__(self, model_explainer, nepali_font=NEPALI_FONT):
        self.model_explainer = model_explainer
        self.nepali_font = nepali_font
        self.explainer = shap.Explainer(model_explainer.predict_proba,
                                        shap.maskers.Text(model_explainer.tokenizer))
    
    def explain_and_visualize(self, text):
        sv = self.explainer([text])[0]
        tokens = list(sv.data)
        values_array = np.array(sv.values)
        pred_probs = self.model_explainer.predict_proba([text])[0]
        class_idx = int(np.argmax(pred_probs))
        token_values = values_array if values_array.ndim == 1 else values_array[:, class_idx] if values_array.ndim == 2 else values_array[0, :, class_idx]
        max_len = min(len(tokens), len(token_values))
        tokens = tokens[:max_len]
        token_values = token_values[:max_len]
        model_tokens, display_tokens = build_token_alignment(text)
        word_scores, token_ptr = [], 0
        for model_tok, disp_tok in zip(model_tokens, display_tokens):
            score, consumed = 0.0, ""
            while token_ptr < max_len and len(consumed) < len(model_tok):
                score += float(token_values[token_ptr])
                consumed += tokens[token_ptr].replace("▁", "")
                token_ptr += 1
            word_scores.append((disp_tok, score))
        if not word_scores:
            print("⚠️ Empty SHAP attribution")
            return
        max_val = max(abs(v) for _, v in word_scores) + 1e-6
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(word_scores)), 2.4))
        ax.axis("off")
        x, y, text_objs = 0.01, 0.5, []
        for word, val in word_scores:
            intensity = min(abs(val) / max_val, 1.0)
            color = (1.0, 1.0 - intensity, 1.0 - intensity)
            txt = ax.text(x, y, f" {word} ", fontsize=13,
                          bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, boxstyle="round,pad=0.3"))
            text_objs.append(txt)
            x += 0.04 + 0.028 * len(word)
            if x > 0.95:
                x, y = 0.01, y - 0.45
        apply_nepali_font(ax, self.nepali_font, texts=text_objs, is_tick_labels=False)
        out_path = os.path.join(OUT_DIR, f"shap_explanation_{self.model_explainer.class_names[class_idx]}_{abs(hash(text)) % 10**8}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✅ SHAP image saved to: {out_path}")

# ------------------------- Execution ---------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nepali Hate Speech Explainability")
    parser.add_argument("--local_model_path", type=str, default=None, help="Local model path")
    parser.add_argument("--hf_model_id", type=str, default="UDHOV/xlm-roberta-large-nepali-hate-classification", help="HuggingFace model ID")
    parser.add_argument("--test_data", type=str, default="test.json", help="Path to test data")
    args = parser.parse_args()

    model, tokenizer, le = load_model(args.local_model_path, args.hf_model_id)
    explainer = XLMRobertaExplainer(model, tokenizer, le)

    df = pd.read_json(args.test_data)
    sampled_df = df.groupby("Label_Multiclass", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 3), random_state=42))

    for i, (_, row) in enumerate(sampled_df.iterrows(), start=1):
        text = str(row["Comment"])
        true_label = row.get("Label_Multiclass", "N/A")
        analysis = explainer.predict_with_analysis(text)
        print(f"\nSample {i}/{len(sampled_df)}")
        print(f"Text: {text}")
        print(f"True Label: {true_label}")
        print(f"Prediction: {analysis['predicted_label']}")
        print(f"Confidence: {analysis['confidence']:.4f}")

        if LIME_AVAILABLE:
            print("\n--- LIME Explanation ---")
            LIMEExplainer(explainer).explain_and_visualize(text)

        if SHAP_AVAILABLE:
            print("\n--- SHAP Explanation ---")
            SHAPExplainer(explainer).explain_and_visualize(text)
