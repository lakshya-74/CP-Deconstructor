import json
from old_basic.step3_constraint_parser import run_module_2_baseline

# =====================================================================
# MODULE 3 BASELINE: Score & Rank Plausible Algorithmic Patterns
# =====================================================================
def predict_algorithmic_patterns(m2_output):
    """
    Combines extracted text entities + complexity constraints (Fusion!)
    to score and rank plausible algorithmic patterns with clear evidence.
    """
    task_text = m2_output["simplified_task"].lower()
    entities = [e["entity_text"] for e in m2_output["extracted_entities"]]
    comp_guide = m2_output["complexity_guidance"]
    max_n = comp_guide.get("max_size", 0)

    # Check if O(n^2) is marked as LIKELY TLE by Module 2
    n2_is_too_slow = False
    for row in comp_guide.get("complexity_table", []):
        if row["complexity"] == "O(n^2)" and "TLE" in row["verdict"]:
            n2_is_too_slow = True

    candidates = []

    # --- PATTERN 1: Hashing / Frequency Map ---
    score = 0
    reasons = []
    if any(k in entities for k in ["two distinct elements", "pair"]) and "add up to" in entities:
        score += 50
        reasons.append("Detected pair-sum structure ('two distinct elements' + 'add up to')")
    if "distinct" in task_text or "frequency" in task_text:
        score += 20
        reasons.append("Problem mentions element uniqueness/frequency")
    if n2_is_too_slow:
        score += 20
        reasons.append(f"n <= {max_n} makes O(n^2) pair-checking too slow; O(1) hash lookup fits O(n)")
    if score > 0:
        candidates.append({"pattern": "hashing", "score": score, "evidence": reasons})

    # --- PATTERN 2: Sorting + Two Pointers ---
    score = 0
    reasons = []
    if any(k in entities for k in ["two distinct elements", "pair", "contiguous subarray"]):
        score += 40
        reasons.append("Detected pair or contiguous segment condition")
    if "add up to" in entities or "sum" in entities:
        score += 25
        reasons.append("Target sum condition works well with sorted two-pointer shrinking")
    if n2_is_too_slow:
        score += 20
        reasons.append(f"Sorting + Two Pointers runs in O(n log n), which passes for n <= {max_n}")
    if score > 0:
        candidates.append({"pattern": "two_pointers", "score": score, "evidence": reasons})

    # --- PATTERN 3: Binary Search ---
    score = 0
    reasons = []
    if "sorted" in entities or "non-decreasing" in entities:
        score += 50
        reasons.append("Input is already ordered ('sorted' / 'non-decreasing')")
    if "minimize the maximum" in task_text or "maximize the minimum" in task_text:
        score += 60
        reasons.append("Classic 'min-max' phrase strongly signals Binary Search on Answer")
    if "add up to" in entities and n2_is_too_slow:
        score += 25
        reasons.append("After sorting, you can binary search for the complement (T - a[i]) in O(n log n)")
    if score > 0:
        candidates.append({"pattern": "binary_search", "score": score, "evidence": reasons})

    # --- PATTERN 4: Dynamic Programming (DP) ---
    score = 0
    reasons = []
    if ("minimum" in entities or "maximum" in entities or "count the number" in task_text) and "subsequence" in entities:
        score += 55
        reasons.append("Optimization/counting over subsequences is a classic DP signal")
    if 0 < max_n <= 5000 and not n2_is_too_slow:
        score += 25
        reasons.append(f"Constraint n <= {max_n} comfortably allows an O(n^2) DP state table")
    if score > 0:
        candidates.append({"pattern": "dynamic_programming", "score": score, "evidence": reasons})

    # --- PATTERN 5: Graphs / Trees (DFS / BFS) ---
    score = 0
    reasons = []
    if any(k in entities for k in ["tree", "graph", "connected", "undirected", "directed", "shortest path"]):
        score += 70
        reasons.append("Explicit graph/tree topology entities found in problem statement")
    if score > 0:
        candidates.append({"pattern": "graphs_trees", "score": score, "evidence": reasons})

    # --- PATTERN 6: Greedy ---
    score = 0
    reasons = []
    if ("minimum" in entities or "maximum" in entities) and n2_is_too_slow:
        score += 30
        reasons.append(f"Optimization goal with large bound (n <= {max_n}) often requires O(n log n) greedy sorting")
    if score > 0:
        candidates.append({"pattern": "greedy", "score": score, "evidence": reasons})

    # Sort patterns from highest score to lowest
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Fallback if no specific pattern matched strongly
    if not candidates:
        candidates.append({
            "pattern": "general_implementation",
            "score": 10,
            "evidence": ["Uncertain pattern: no strong keyword triggers matched; inspect constraints manually."]
        })

    return candidates[:3]  # Return top 3 ranked possibilities


# =====================================================================
# MODULE 4 BASELINE: Deterministic Progressive Hint Assembler
# =====================================================================
def assemble_progressive_hints(m2_output, ranked_patterns):
    """
    Uses verified facts + top predicted patterns to create 3 progressive hints
    (from general problem framing to specific algorithmic intuition) without leaking code.
    """
    simplified = m2_output["simplified_task"]
    comp = m2_output["complexity_guidance"]
    primary_var = comp.get("primary_variable", "n")
    max_n = comp.get("max_size", None)
    top_pattern = ranked_patterns[0]["pattern"] if ranked_patterns else "general_implementation"

    hints = []

    # --- HINT 1: Strip the story & highlight the complexity bottleneck ---
    if max_n and max_n >= 10000:
        hint1_text = (
            f"Focus only on the core task: \"{simplified}\" "
            f"Notice that ${primary_var} \\le {max_n:,}$. A brute-force approach that checks all pairs "
            f"or nested loops takes $O({primary_var}^2) \\approx {max_n**2:,}$ operations, which will likely TLE. "
            f"You need an $O({primary_var})$ or $O({primary_var} \\log {primary_var})$ approach."
        )
        hint1_evidence = f"Grounded in extracted bound {primary_var} <= {max_n} and simplified task."
    else:
        hint1_text = f"Strip away the story and focus on the core task: \"{simplified}\""
        hint1_evidence = "Grounded in Module 1 discourse role tagging (INPUT_SPEC + TASK)."

    hints.append({
        "stage": 1,
        "level": "Problem Framing & Complexity Check",
        "hint": hint1_text,
        "provenance": hint1_evidence
    })

    # --- HINT 2: Guide the student toward the right algorithmic family ---
    pattern_names = [p["pattern"] for p in ranked_patterns[:2]]
    hint2_text = (
        f"Instead of nested loops, consider {', '.join(pattern_names)}. "
        f"Why? {ranked_patterns[0]['evidence'][0]}."
    )
    hints.append({
        "stage": 2,
        "level": "Algorithmic Direction",
        "hint": hint2_text,
        "provenance": f"Triggered by top ranked pattern '{top_pattern}' clues."
    })

    # --- HINT 3: Specific state/complement intuition (Still zero code!) ---
    hint3_templates = {
        "hashing": (
            "As you iterate through each element x in the array, ask yourself: "
            "'What exact partner value do I need to reach the target T?' (Hint: T - x). "
            "Store previously visited elements in a Hash Set / Map so you can check if that partner already exists in O(1) time."
        ),
        "two_pointers": (
            "Try sorting the array first in O(n log n) time. Place one pointer at the smallest element (left) "
            "and one at the largest element (right). Compare their sum to T to decide which pointer to move inward."
        ),
        "binary_search": (
            "If the search space is monotonic (sorted), you don't need to scan linearly. "
            "Use Binary Search to cut the search range in half at each step in O(log n) time."
        ),
        "dynamic_programming": (
            "Break the problem into smaller prefix subproblems. Define what state dp[i] represents for the first i elements, "
            "and write how dp[i] transitions from previous states."
        ),
        "graphs_trees": (
            "Model the entities as vertices and their connections as edges. Consider whether a DFS/BFS traversal "
            "can propagate the required information from neighbors or children."
        ),
        "greedy": (
            "Try sorting the input by a key property (like end time, weight, or value) and see if making the locally optimal "
            "choice at each step stays safe globally."
        )
    }

    hint3_text = hint3_templates.get(
        top_pattern,
        "Try working through a small 4-element example by hand and observe what redundant work your brute-force solution repeats."
    )

    hints.append({
        "stage": 3,
        "level": "Core Algorithmic Trick (No Code Spoiler)",
        "hint": hint3_text,
        "provenance": f"Template grounded in verified pattern '{top_pattern}'."
    })

    return hints


# =====================================================================
# COMPLETE END-TO-END MVP PIPELINE (MODULES 1 -> 2 -> 3 -> 4)
# =====================================================================
def run_full_mvp_pipeline(problem_text):
    # Run Module 1 & Module 2
    m2_data = run_module_2_baseline(problem_text)

    # Run Module 3 (Pattern Ranking)
    ranked_patterns = predict_algorithmic_patterns(m2_data)

    # Run Module 4 (Progressive Hint Assembly)
    progressive_hints = assemble_progressive_hints(m2_data, ranked_patterns)

    return {
        "1_simplified_task": m2_data["simplified_task"],
        "2_extracted_entities": m2_data["extracted_entities"],
        "3_parsed_constraints": m2_data["parsed_constraints"],
        "4_complexity_guidance": m2_data["complexity_guidance"],
        "5_ranked_algorithmic_patterns": ranked_patterns,
        "6_progressive_hints": progressive_hints
    }


# --- TEST THE ENTIRE END-TO-END SYSTEM ---
if __name__ == "__main__":
    sample_problem = (
        "Alice and Bob are visiting a magical park with many attractions. "
        "You are given an array of $n$ integers $a_1, a_2, ..., a_n$ representing ticket prices. "
        "Determine whether two distinct elements add up to a target $T$. "
        "It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i, T <= 10^9$."
    )

    full_result = run_full_mvp_pipeline(sample_problem)
    print(json.dumps(full_result, indent=2))