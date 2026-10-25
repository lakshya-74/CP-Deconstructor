import os
import json
import sys
import joblib
import numpy as np
import sklearn_crfsuite
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# Ensure Python always finds the main project folder ('cp_deconstructor')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.preprocessing import preprocess_and_split_sentences
from src.discourse_tagger import (
    CRF_MODEL_PATH,
    PRETRAINED_CLASSIFIER_PATH,
    PRETRAINED_ENCODER_NAME,
    extract_crf_features,
    build_pretrained_feature_vector,
    predict_role_rules_baseline
)

ALL_ROLES = ["STORY", "INPUT_SPEC", "TASK", "CONSTRAINT", "EXAMPLE"]


def load_json_dataset(filepath):
    """Loads our JSON dataset and runs preprocessing on each sentence."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_problems = json.load(f)

    processed_problems = []
    for prob in raw_problems:
        sent_infos = []
        labels = []
        for idx, item in enumerate(prob["sentences"]):
            info = preprocess_and_split_sentences(item["text"])[0]
            info["sentence_id"] = idx
            sent_infos.append(info)
            labels.append(item["role"])
        processed_problems.append((prob["problem_id"], sent_infos, labels))
    return processed_problems


def compute_macro_f1(y_true_flat, y_pred_flat):
    """Calculates Overall Accuracy, per-role F1, and Macro-F1."""
    stats = {r: {"TP": 0, "FP": 0, "FN": 0} for r in ALL_ROLES}
    correct = 0
    for t, p in zip(y_true_flat, y_pred_flat):
        if t == p:
            stats[t]["TP"] += 1
            correct += 1
        else:
            stats[p]["FP"] += 1
            stats[t]["FN"] += 1

    f1_list = []
    for r in ALL_ROLES:
        tp, fp, fn = stats[r]["TP"], stats[r]["FP"], stats[r]["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    accuracy = correct / len(y_true_flat)
    macro_f1 = sum(f1_list) / len(f1_list)
    return accuracy, macro_f1


def main():
    os.makedirs("saved_models", exist_ok=True)
    train_problems = load_json_dataset("data/train_sentences.json")
    test_problems = load_json_dataset("data/gold_test_set.json")

    # ---------------------------------------------------------
    # 1. TRAIN & SAVE THE CRF MODEL
    # ---------------------------------------------------------
    X_train_crf = [[extract_crf_features(sents, i) for i in range(len(sents))] for _, sents, _ in train_problems]
    y_train_crf = [labels for _, _, labels in train_problems]

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train_crf, y_train_crf)
    joblib.dump(crf, CRF_MODEL_PATH)
    print(f"[Saved] Classical CRF model -> {CRF_MODEL_PATH}")

    # ---------------------------------------------------------
    # 2. TRAIN & SAVE THE PRETRAINED TRANSFORMER CLASSIFIER
    # ---------------------------------------------------------
    print(f"[Loading] Pretrained Hugging Face Encoder: {PRETRAINED_ENCODER_NAME}...")
    encoder = SentenceTransformer(PRETRAINED_ENCODER_NAME)

    X_train_vec = np.vstack([build_pretrained_feature_vector(encoder, sents) for _, sents, _ in train_problems])
    y_train_flat = [role for _, _, labels in train_problems for role in labels]

    clf = LogisticRegression(C=2.0, max_iter=500)
    clf.fit(X_train_vec, y_train_flat)
    joblib.dump(clf, PRETRAINED_CLASSIFIER_PATH)
    print(f"[Saved] Pretrained Transformer Classifier -> {PRETRAINED_CLASSIFIER_PATH}\n")

    # ---------------------------------------------------------
    # 3. EVALUATE ALL 3 MODELS ON THE GOLD TEST SET (19 Sentences)
    # ---------------------------------------------------------
    y_true_all = [role for _, _, labels in test_problems for role in labels]

    # Model A: Rule-Based Baseline
    y_pred_rules = [
        predict_role_rules_baseline(s, len(sents))[0]
        for _, sents, _ in test_problems for s in sents
    ]

    # Model B: Trained CRF
    X_test_crf = [[extract_crf_features(sents, i) for i in range(len(sents))] for _, sents, _ in test_problems]
    y_pred_crf = [role for seq in crf.predict(X_test_crf) for role in seq]

    # Model C: Pretrained Transformer (all-MiniLM-L6-v2)
    X_test_vec = np.vstack([build_pretrained_feature_vector(encoder, sents) for _, sents, _ in test_problems])
    y_pred_pretrained = list(clf.predict(X_test_vec))

    # Print Ablation / Comparison Table
    print("=" * 75)
    print(" MODULE 1 ABLATION TABLE: RULES vs. CRF vs. PRETRAINED TRANSFORMER")
    print("=" * 75)
    print(f"{'MODEL ARCHITECTURE':<38} | {'ACCURACY':<15} | {'MACRO-F1':<10}")
    print("-" * 75)

    for name, preds in [
        ("1. Naive Rule-Based Baseline", y_pred_rules),
        ("2. Classical Linear-Chain CRF", y_pred_crf),
        ("3. Pretrained Transformer (MiniLM)", y_pred_pretrained)
    ]:
        acc, mf1 = compute_macro_f1(y_true_all, preds)
        correct_cnt = sum(1 for t, p in zip(y_true_all, preds) if t == p)
        print(f"{name:<38} | {acc:>6.2%} ({correct_cnt}/{len(y_true_all)}) | {mf1:<10.2f}")
    print("=" * 75)

if __name__ == "__main__":
    main()