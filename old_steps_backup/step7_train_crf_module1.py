import re
import sklearn_crfsuite
from old_basic.step1_preprocessor import protect_latex_and_split_sentences
from old_basic.step6_evaluate_module1 import gold_test_problems

# ==============================================================================
# 1. FEATURE EXTRACTOR: Turn each sentence into a dictionary of smart clues
# ==============================================================================
def has_real_variable(math_spans):
    """
    Returns True ONLY if a math span contains letters/variables (like $n$, $a_i$, $k$),
    ignoring plain story numbers like $2050$, $3$, or $1$.
    """
    for span in math_spans:
        clean = span.strip("$")
        # Remove LaTeX commands like \le, \ge, \dots, \times, \cdot, \sum
        clean = re.sub(r'\\[a-zA-Z]+', '', clean)
        # If any English letter (a-z, A-Z) remains, it's a real algebraic variable!
        if re.search(r'[a-zA-Z]', clean):
            return True
    return False


def extract_sentence_features(problem_sentences_info, idx):
    """
    Extracts clues (features) for sentence at position `idx` inside a problem.
    Notice how we also include clues from the PREVIOUS and NEXT sentence!
    """
    total = len(problem_sentences_info)
    sent_info = problem_sentences_info[idx]
    orig = sent_info["original_sentence"]
    lower = orig.lower()
    math_spans = sent_info["extracted_math_spans"]
    words = re.findall(r'[a-zA-Z]+', lower)
    first_word = words[0] if words else ""

    # Check what kind of math is inside this sentence
    real_var = has_real_variable(math_spans)
    only_numbers = (len(math_spans) > 0) and (not real_var)

    # Check for inequality symbols OR verbal constraint phrases ("does not exceed", "at most")
    inequality_regex = r'(<=|>=|\\le|\\ge|\\leq|\\geq|<|>|10\^)'
    has_math_ineq = any(re.search(inequality_regex, m) for m in math_spans)
    verbal_bound_phrases = [
        "guaranteed that", "constraints", "does not exceed", "at most",
        "no more than", "at least", "sum of", "over all test cases"
    ]
    has_verbal_bound = any(p in lower for p in verbal_bound_phrases)

    # Check for Input Specification phrases
    input_phrases = [
        "you are given", "given an", "given a", "consists of", "consisting of",
        "contains", "array", "string", "tree", "graph", "cities", "connected by",
        "represented by", "denotes"
    ]
    has_input_cue = any(p in lower for p in input_phrases)

    # Check for Task / Goal phrases and imperative starting verbs
    task_phrases = [
        "determine whether", "find the", "calculate the", "compute the",
        "output the", "print the", "what is the", "count the", "return the",
        "your goal is", "minimum", "maximum", "longest", "shortest", "possible"
    ]
    has_task_cue = any(p in lower for p in task_phrases) or orig.endswith("?")
    imperative_verbs = {"determine", "find", "calculate", "compute", "output", "print", "count", "return"}
    starts_with_command = (first_word in imperative_verbs)

    # Check for Story / Past-tense narrative clues
    story_phrases = [
        "in the year", "once upon", "wanted to", "bought", "visiting",
        "magical", "kingdom", "always", "secret treasure", "playing a game",
        "decided to", "loves", "birthday"
    ]
    has_story_cue = any(p in lower for p in story_phrases)

    # Check for Example clues
    example_phrases = ["for example", "in the first test case", "sample", "explanation:", "the answer is"]
    has_example_cue = any(p in lower for p in example_phrases)

    # Build the feature dictionary for this sentence
    features = {
        "bias": 1.0,
        "is_first": (idx == 0),
        "is_last": (idx == total - 1),
        "position_ratio": round(idx / max(1, total - 1), 1),
        "has_real_variable": real_var,
        "has_only_plain_numbers": only_numbers,
        "has_bound_or_inequality": (has_math_ineq or has_verbal_bound),
        "has_input_cue": has_input_cue,
        "has_task_cue": has_task_cue,
        "starts_with_command": starts_with_command,
        "has_story_cue": has_story_cue,
        "has_example_cue": has_example_cue,
    }

    # Add context from the PREVIOUS sentence (if not the first sentence)
    if idx > 0:
        prev_lower = problem_sentences_info[idx - 1]["original_sentence"].lower()
        prev_math = problem_sentences_info[idx - 1]["extracted_math_spans"]
        features["prev_had_real_var"] = has_real_variable(prev_math)
        features["prev_had_input_cue"] = any(p in prev_lower for p in input_phrases)
        features["prev_had_task_cue"] = any(p in prev_lower for p in task_phrases)
    else:
        features["BOS"] = True  # Beginning Of Statement marker

    # Add context from the NEXT sentence (if not the last sentence)
    if idx < total - 1:
        next_lower = problem_sentences_info[idx + 1]["original_sentence"].lower()
        next_math = problem_sentences_info[idx + 1]["extracted_math_spans"]
        features["next_has_inequality"] = any(re.search(inequality_regex, m) for m in next_math) or any(p in next_lower for p in verbal_bound_phrases)
    else:
        features["EOS"] = True  # End Of Statement marker

    return features


# ==============================================================================
# 2. TRAINING DATASET (Separate from our Gold Test Set!)
# ==============================================================================
train_problems = [
    [
        ("Monocarp is playing a video game where he collects magical coins.", "STORY"),
        ("You are given an array $a$ consisting of $n$ integers.", "INPUT_SPEC"),
        ("Find the maximum sum of a subarray of length $k$.", "TASK"),
        ("It is guaranteed that $1 <= k <= n <= 10^5$.", "CONSTRAINT"),
        ("For example, if $a = [1, 4, 2, 10]$ and $k = 2$, the answer is $12$.", "EXAMPLE")
    ],
    [
        ("In the year $3000$, Chef bought $5$ robots to clean his restaurant.", "STORY"),
        ("Chef always wanted to find the fastest way to clean all tables.", "STORY"),
        ("The restaurant is represented by a grid of $n$ rows and $m$ columns.", "INPUT_SPEC"),
        ("Your goal is to calculate the minimum time required to visit all dirty cells.", "TASK"),
        ("The sum of $n \\cdot m$ over all test cases does not exceed $200000$.", "CONSTRAINT")
    ],
    [
        ("Polycarp loves strings and decided to invent a new puzzle.", "STORY"),
        ("Given a string $s$ of length $n$ containing lowercase English letters.", "INPUT_SPEC"),
        ("Return the length of the longest palindrome that can be formed.", "TASK"),
        ("Here $1 <= n <= 2 \\cdot 10^5$.", "CONSTRAINT"),
        ("In the first test case, the longest palindrome has length $7$.", "EXAMPLE")
    ],
    [
        ("There are $n$ islands connected by $m$ bridges in a magical archipelago.", "INPUT_SPEC"),
        ("Each bridge has a durability weight $w_i$.", "INPUT_SPEC"),
        ("Determine whether it is possible to travel from island $1$ to island $n$.", "TASK"),
        ("It is guaranteed that $2 <= n <= 10^5$ and $1 <= w_i <= 10^9$.", "CONSTRAINT")
    ],
    [
        ("Alice and Bob are playing a game with $2$ piles of stones.", "STORY"),
        ("Initially, you are given $n$ piles where the $i$-th pile has $c_i$ stones.", "INPUT_SPEC"),
        ("Count the number of winning first moves for Alice.", "TASK"),
        ("The value of $n$ does not exceed $50000$.", "CONSTRAINT"),
        ("For example, when $c = [3, 4, 5]$, Alice has $1$ winning move.", "EXAMPLE")
    ],
    [
        ("During his vacation, Vasya visited $10$ ancient castles.", "STORY"),
        ("He wanted to find the oldest artifact in the museum.", "STORY"),
        ("An array $p$ of $n$ integers denotes the age of each artifact.", "INPUT_SPEC"),
        ("Your goal is to find the minimum cost to buy at least $k$ artifacts.", "TASK"),
        ("Constraints: $1 <= k <= n <= 200000$ and $1 <= p_i <= 10^6$.", "CONSTRAINT")
    ]
]


def prepare_crf_data(labeled_problems):
    """Converts a list of labeled problems into X (feature sequences) and y (label sequences)."""
    X_data, y_data = [], []
    for prob in labeled_problems:
        sent_infos = []
        labels = []
        for idx, (raw_sent, role) in enumerate(prob):
            info = protect_latex_and_split_sentences(raw_sent)[0]
            info["sentence_id"] = idx
            sent_infos.append(info)
            labels.append(role)

        prob_features = [extract_sentence_features(sent_infos, i) for i in range(len(sent_infos))]
        X_data.append(prob_features)
        y_data.append(labels)
    return X_data, y_data


# ==============================================================================
# 3. TRAIN THE CRF & EVALUATE ON OUR STEP 6 GOLD TEST SET
# ==============================================================================
def train_and_evaluate_crf():
    X_train, y_train = prepare_crf_data(train_problems)
    X_test, y_test = prepare_crf_data(gold_test_problems)

    # Initialize and train the Linear-Chain CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,  # L1 regularization (keeps only strong features)
        c2=0.1,  # L2 regularization (prevents overfitting)
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Predict roles on the Gold Test Set
    y_pred = crf.predict(X_test)

    # Calculate Precision, Recall, F1 on Gold Test Set
    all_roles = ["STORY", "INPUT_SPEC", "TASK", "CONSTRAINT", "EXAMPLE"]
    stats = {role: {"TP": 0, "FP": 0, "FN": 0} for role in all_roles}
    errors = []
    total_sents = 0
    total_correct = 0

    for p_idx, (true_seq, pred_seq) in enumerate(zip(y_test, y_pred)):
        for s_idx, (true_role, pred_role) in enumerate(zip(true_seq, pred_seq)):
            total_sents += 1
            raw_sent = gold_test_problems[p_idx][s_idx][0]
            if pred_role == true_role:
                stats[true_role]["TP"] += 1
                total_correct += 1
            else:
                stats[pred_role]["FP"] += 1
                stats[true_role]["FN"] += 1
                errors.append((p_idx + 1, raw_sent, true_role, pred_role))

    print("=" * 72)
    print("   TRAINED CRF MODEL RESULTS ON GOLD TEST SET (Compare with 0.66!)")
    print("=" * 72)
    print(f"{'ROLE LABEL':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'TP / FP / FN'}")
    print("-" * 72)

    f1_scores = []
    for role in all_roles:
        tp, fp, fn = stats[role]["TP"], stats[role]["FP"], stats[role]["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        print(f"{role:<15} | {prec:<10.2f} | {rec:<10.2f} | {f1:<10.2f} | {tp} / {fp} / {fn}")

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print("=" * 72)
    print(f"Baseline Accuracy (Step 6): 64.29%  -->  Trained CRF Accuracy: {total_correct / total_sents:.2%} ({total_correct}/{total_sents})")
    print(f"Baseline Macro-F1 (Step 6): 0.66    -->  Trained CRF Macro-F1: {macro_f1:.2f}")
    print("=" * 72)

    if not errors:
        print("\nZERO BUGS! All 5 tricky sentences from Step 6 were classified correctly!")
    else:
        print(f"\nRemaining Errors ({len(errors)}):")
        for p_num, sent, t_role, p_role in errors:
            print(f"  [Problem {p_num}] '{sent}' -> True: {t_role}, Pred: {p_role}")

    # Show what sequence transitions the CRF learned!
    print("\n=== TOP SEQUENCE TRANSITIONS LEARNED BY THE CRF MODEL ===")
    transitions = sorted(crf.transition_features_.items(), key=lambda x: x[1], reverse=True)
    print("Most Likely Role Sequences (Positive Weights):")
    for (label_from, label_to), weight in transitions[:5]:
        print(f"  {label_from:<12} ---> {label_to:<12} (weight: +{weight:.2f})")

    print("\nLeast Likely Role Sequences (Negative Weights - CRF blocks these!):")
    for (label_from, label_to), weight in transitions[-3:]:
        print(f"  {label_from:<12} ---> {label_to:<12} (weight: {weight:.2f})")

if __name__ == "__main__":
    train_and_evaluate_crf()