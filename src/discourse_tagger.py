import os
import re
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

# Paths where trained models are saved
CRF_MODEL_PATH = os.path.join("saved_models", "crf_discourse.pkl")
PRETRAINED_CLASSIFIER_PATH = os.path.join("saved_models", "pretrained_discourse.pkl")
PRETRAINED_ENCODER_NAME = "all-MiniLM-L6-v2"


# ==============================================================================
# 1. BASELINE TIER: Naive Rule-Based Tagger (For Ablation Comparison)
# ==============================================================================
def predict_role_rules_baseline(sent_info, total_sentences):
    """Our original simple keyword baseline so we can always compare against it."""
    orig = sent_info["original_sentence"]
    lower = orig.lower()
    math_spans = sent_info["extracted_math_spans"]

    ineq_pattern = r'(<=|>=|\\le|\\ge|\\leq|\\geq|<|>|10\^)'
    if any(re.search(ineq_pattern, m) for m in math_spans) or "guaranteed that" in lower:
        return "CONSTRAINT", "Matched inequality math or 'guaranteed that'"

    if any(k in lower for k in ["for example", "in the first test case", "sample input"]):
        return "EXAMPLE", "Matched example phrase"

    for kw in ["determine whether", "find the", "calculate the", "compute the", "output the", "print the", "count the"]:
        if kw in lower:
            return "TASK", f"Matched task keyword '{kw}'"

    for kw in ["you are given", "given an", "given a", "consists of", "represented by"]:
        if kw in lower:
            return "INPUT_SPEC", f"Matched input keyword '{kw}'"

    if len(math_spans) > 0:
        return "INPUT_SPEC", "Fallback rule: contains math span"

    return "STORY", "Fallback rule: no math or keywords found"


# ==============================================================================
# 2. CLASSICAL ML TIER: Linear-Chain CRF Feature Extractor
# ==============================================================================
def extract_crf_features(problem_sents, idx):
    """Extracts sequence and structural features for the CRF model."""
    total = len(problem_sents)
    sent_info = problem_sents[idx]
    orig = sent_info["original_sentence"]
    lower = orig.lower()
    math_spans = sent_info["extracted_math_spans"]
    words = re.findall(r'[a-zA-Z]+', lower)
    first_word = words[0] if words else ""

    real_var = sent_info["has_real_variable"]
    only_numbers = (len(math_spans) > 0) and (not real_var)

    ineq_regex = r'(<=|>=|\\le|\\ge|\\leq|\\geq|<|>|10\^)'
    has_math_ineq = any(re.search(ineq_regex, m) for m in math_spans)
    verbal_bounds = ["guaranteed that", "constraints", "does not exceed", "at most", "no more than", "sum of", "note that"]
    has_bound = has_math_ineq or any(p in lower for p in verbal_bounds)

    input_cues = ["you are given", "given an", "given a", "consists of", "consisting of", "contains", "array", "string", "tree", "grid", "sequence", "represents", "denotes", "connected by"]
    task_cues = ["determine whether", "find the", "calculate the", "compute the", "output the", "print the", "count the", "return the", "your goal is", "minimum", "maximum", "longest", "shortest"]
    story_cues = ["in the year", "once upon", "wanted to", "hoped to", "bought", "visiting", "magical", "kingdom", "always", "playing a game", "decided to", "loves", "piloted", "vacation"]
    example_cues = ["for example", "in the first test case", "sample", "explanation:", "the answer is"]

    imperative_verbs = {"determine", "find", "calculate", "compute", "output", "print", "count", "return"}

    features = {
        "bias": 1.0,
        "is_first": (idx == 0),
        "is_last": (idx == total - 1),
        "pos_ratio": round(idx / max(1, total - 1), 1),
        "has_real_var": real_var,
        "only_numbers": only_numbers,
        "has_bound": has_bound,
        "has_input_cue": any(p in lower for p in input_cues),
        "has_task_cue": any(p in lower for p in task_cues) or orig.endswith("?"),
        "starts_with_command": (first_word in imperative_verbs),
        "has_story_cue": any(p in lower for p in story_cues),
        "has_example_cue": any(p in lower for p in example_cues),
    }

    if idx > 0:
        features["prev_real_var"] = problem_sents[idx - 1]["has_real_variable"]
    else:
        features["BOS"] = True

    if idx < total - 1:
        next_math = problem_sents[idx + 1]["extracted_math_spans"]
        features["next_has_ineq"] = any(re.search(ineq_regex, m) for m in next_math)
    else:
        features["EOS"] = True

    return features


# ==============================================================================
# 3. PRETRAINED TRANSFORMER TIER: Contextual Embeddings + Structural Features
# ==============================================================================
def build_pretrained_feature_vector(encoder, problem_sents):
    """
    Uses a Hugging Face Pretrained Transformer ('all-MiniLM-L6-v2') to convert each
    protected sentence into a 384-dimensional contextual embedding, and fuses it with
    5 structural math/position features so the model understands both English & Math!
    """
    # Use protected_view (__MATH_0__) so raw LaTeX symbols don't confuse the English encoder
    texts = [s["protected_view"] for s in problem_sents]
    embeddings = encoder.encode(texts, show_progress_bar=False)

    total = len(problem_sents)
    ineq_regex = r'(<=|>=|\\le|\\ge|\\leq|\\geq|<|>|10\^)'
    verbal_bounds = ["guaranteed that", "constraints", "does not exceed", "at most", "note that"]

    combined_vectors = []
    for idx, sent_info in enumerate(problem_sents):
        lower = sent_info["original_sentence"].lower()
        math_spans = sent_info["extracted_math_spans"]

        pos_ratio = idx / max(1, total - 1)
        real_var = 1.0 if sent_info["has_real_variable"] else 0.0
        only_num = 1.0 if (len(math_spans) > 0 and not sent_info["has_real_variable"]) else 0.0
        has_ineq = 1.0 if (any(re.search(ineq_regex, m) for m in math_spans) or any(b in lower for b in verbal_bounds)) else 0.0
        has_example = 1.0 if any(e in lower for e in ["for example", "in the first test case", "the answer is"]) else 0.0

        # Scale structural features slightly so they balance with the 384 embedding numbers
        struct_feats = np.array([pos_ratio, real_var, only_num, has_ineq, has_example]) * 1.5
        fused_vec = np.concatenate([embeddings[idx], struct_feats])
        combined_vectors.append(fused_vec)

    return np.array(combined_vectors)


# ==============================================================================
# 4. UNIFIED CONTROLLER: Train, Load, and Predict with Any of the 3 Models
# ==============================================================================
class DiscourseRoleTagger:
    def __init__(self, mode="pretrained"):
        """
        mode can be: 'rules', 'crf', or 'pretrained'
        """
        self.mode = mode
        self.crf_model = None
        self.pretrained_clf = None
        self.encoder = None

        if mode == "crf" and os.path.exists(CRF_MODEL_PATH):
            self.crf_model = joblib.load(CRF_MODEL_PATH)
        elif mode == "pretrained":
            self.encoder = SentenceTransformer(PRETRAINED_ENCODER_NAME)
            if os.path.exists(PRETRAINED_CLASSIFIER_PATH):
                self.pretrained_clf = joblib.load(PRETRAINED_CLASSIFIER_PATH)

    def tag_problem_sentences(self, problem_text):
        """Splits a problem into sentences, tags their roles, and returns the Simplified Task."""
        sents = preprocess_and_split_sentences(problem_text)
        if not sents:
            return {"simplified_task": "", "tagged_sentences": []}

        if self.mode == "rules":
            for s in sents:
                role, evidence = predict_role_rules_baseline(s, len(sents))
                s["role"] = role
                s["evidence"] = evidence

        elif self.mode == "crf":
            feats = [extract_crf_features(sents, i) for i in range(len(sents))]
            preds = self.crf_model.predict_single(feats)
            for s, role in zip(sents, preds):
                s["role"] = role
                s["evidence"] = "Predicted by trained Linear-Chain CRF sequence model"

        elif self.mode == "pretrained":
            X_vec = build_pretrained_feature_vector(self.encoder, sents)
            preds = self.pretrained_clf.predict(X_vec)
            probs = self.pretrained_clf.predict_proba(X_vec)
            max_conf = np.max(probs, axis=1)
            for s, role, conf in zip(sents, preds, max_conf):
                s["role"] = str(role)
                s["confidence"] = round(float(conf), 3)
                s["evidence"] = f"Predicted by Pretrained Transformer ({PRETRAINED_ENCODER_NAME}) with {conf:.1%} confidence"

        essential = [s["original_sentence"] for s in sents if s["role"] in ("INPUT_SPEC", "TASK")]
        return {
            "model_used": self.mode,
            "simplified_task": " ".join(essential),
            "tagged_sentences": sents
        }