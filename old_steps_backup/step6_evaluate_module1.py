from old_basic.step2_role_tagger import classify_sentence_role
from old_basic.step1_preprocessor import protect_latex_and_split_sentences

# ==============================================================================
# 1. GOLD BENCHMARK DATASET (With Tricky Sentences to Test for Bugs!)
# Each item has the raw sentence and its 100% true human label ("ground_truth")
# ==============================================================================
gold_test_problems = [
    # Problem 1: Standard problem (Our baseline should get 100% here)
    [
        ("Alice and Bob are visiting a magical park.", "STORY"),
        ("You are given an array of $n$ integers $a_1, a_2, ..., a_n$.", "INPUT_SPEC"),
        ("Determine whether two distinct elements add up to $T$.", "TASK"),
        ("It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i <= 10^9$.", "CONSTRAINT"),
        ("For example, if $a = [2, 7, 11]$ and $T = 9$, the answer is YES.", "EXAMPLE")
    ],
    # Problem 2: Tricky Story with Math Numbers + Subtle Task Wording
    [
        ("In the year $2050$, Byteasar bought $3$ magical swords for his kingdom.", "STORY"),
        ("The kingdom consists of $n$ cities connected by $m$ bidirectional roads.", "INPUT_SPEC"),
        ("Your goal is to make all cities reachable from city $1$ using the minimum total cost.", "TASK"),
        ("Here $1 \\le n, m \\le 10^5$.", "CONSTRAINT")
    ],
    # Problem 3: Tricky Constraint without '<=' symbol + Story with fake Task word
    [
        ("Bob always wanted to find the secret treasure hidden in the mountains.", "STORY"),
        ("An array $b$ of length $n$ contains the heights of the mountains.", "INPUT_SPEC"),
        ("Return the length of the longest contiguous subarray with sum equal to $k$.", "TASK"),
        ("The sum of $n$ over all test cases does not exceed $200000$.", "CONSTRAINT"),
        ("In the first test case, the subarray $[1, 3]$ has sum $5$.", "EXAMPLE")
    ]
]


# ==============================================================================
# 2. METRICS CALCULATOR (Precision, Recall, F1-Score from scratch!)
# ==============================================================================
def evaluate_module1_baseline():
    all_roles = ["STORY", "INPUT_SPEC", "TASK", "CONSTRAINT", "EXAMPLE"]
    
    # Counters for True Positives (TP), False Positives (FP), False Negatives (FN)
    stats = {role: {"TP": 0, "FP": 0, "FN": 0} for role in all_roles}
    errors = []
    total_sentences_tested = 0
    total_correct = 0

    for prob_idx, problem_sentences in enumerate(gold_test_problems):
        num_sents = len(problem_sentences)
        
        for sent_idx, (raw_sent, true_role) in enumerate(problem_sentences):
            total_sentences_tested += 1
            
            # Use our Step 1 function on this sentence to get its math spans
            step1_info = protect_latex_and_split_sentences(raw_sent)[0]
            step1_info["sentence_id"] = sent_idx
            
            # Run our Step 2 baseline function to get the predicted role
            prediction_dict = classify_sentence_role(step1_info, num_sents)
            pred_role = prediction_dict["role"]

            # Compare Prediction vs Ground Truth
            if pred_role == true_role:
                stats[true_role]["TP"] += 1
                total_correct += 1
            else:
                stats[pred_role]["FP"] += 1  # False alarm for pred_role
                stats[true_role]["FN"] += 1  # Missed target for true_role
                errors.append({
                    "problem_num": prob_idx + 1,
                    "sentence": raw_sent,
                    "true_role": true_role,
                    "predicted_role": pred_role,
                    "wrong_reason": prediction_dict["evidence"]
                })

    # --- PRINT PRECISION, RECALL, AND F1 TABLE ---
    print("=" * 72)
    print(f"{'ROLE LABEL':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'TP / FP / FN'}")
    print("=" * 72)

    f1_scores = []
    for role in all_roles:
        tp = stats[role]["TP"]
        fp = stats[role]["FP"]
        fn = stats[role]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        print(f"{role:<15} | {precision:<10.2f} | {recall:<10.2f} | {f1:<10.2f} | {tp} / {fp} / {fn}")

    macro_f1 = sum(f1_scores) / len(f1_scores)
    accuracy = total_correct / total_sentences_tested
    print("=" * 72)
    print(f"Overall Accuracy: {accuracy:.2%} ({total_correct}/{total_sentences_tested} sentences correct)")
    print(f"Macro-F1 Score:   {macro_f1:.2f}\n")

    # --- PRINT ERROR ANALYSIS (EXPOSING THE BUGS!) ---
    print(f"=== ERROR ANALYSIS: FOUND {len(errors)} BUGS IN MODULE 1 BASELINE ===")
    for i, err in enumerate(errors, 1):
        print(f"\nBug #{i} (Problem {err['problem_num']}):")
        print(f"  Sentence:       \"{err['sentence']}\"")
        print(f"  True Role:      {err['true_role']}")
        print(f"  Predicted Role: {err['predicted_role']}")
        print(f"  Why it failed:  {err['wrong_reason']}")

if __name__ == "__main__":
    evaluate_module1_baseline()