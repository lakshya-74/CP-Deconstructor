import re
import math
import json
from old_basic.step2_role_tagger import run_module_1_baseline

# ==========================================================
# JOB 1: Convert messy LaTeX numbers ("2 * 10^5") to integers
# ==========================================================
def parse_math_number(num_str):
    """
    Turns CP number strings like '200,000', '10^9', '2 * 10^5', or '2 \\cdot 10^5'
    into an actual Python integer (e.g., 200000).
    """
    s = num_str.strip().replace(",", "").replace(" ", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")

    # Case 1: Multiplier with power of 10, like "2*10^5" or "1.5*10^6"
    match_mult = re.match(r'^(\d+(?:\.\d+)?)\*10\^\{?(\d+)\}?$', s)
    if match_mult:
        coeff = float(match_mult.group(1))
        exp = int(match_mult.group(2))
        return int(coeff * (10 ** exp))

    # Case 2: Simple power of 10, like "10^9" or "10^{18}"
    match_pow = re.match(r'^10\^\{?(\d+)\}?$', s)
    if match_pow:
        exp = int(match_pow.group(1))
        return 10 ** exp

    # Case 3: Plain integer like "200000" or "1"
    if s.lstrip('-').isdigit():
        return int(s)

    return None


# ==========================================================
# JOB 2: Extract variable bounds from math formulas
# ==========================================================
def extract_constraints_from_math(math_spans):
    """
    Reads math expressions like '$1 <= n <= 200000$' or '$1 <= a_i, T <= 10^9$'
    and outputs structured records linking each variable to its min and max.
    """
    extracted_bounds = []

    for span in math_spans:
        # Remove dollar signs and normalize LaTeX <= to standard <=
        clean = span.strip("$").replace("\\le", "<=").replace("\\leq", "<=")
        clean = clean.replace("\\ge", ">=").replace("\\geq", ">=")

        # Pattern A: Chained inequality like "1 <= n <= 200000" or "1 <= a_i, T <= 10^9"
        chain_match = re.search(r'([-\d\.\*\^\\a-zA-Z,]+)\s*<=\s*([a-zA-Z0-9_,\s]+)\s*<=\s*([-\d\.\*\^\\a-zA-Z,]+)', clean)
        if chain_match:
            raw_min, raw_vars, raw_max = chain_match.groups()
            min_val = parse_math_number(raw_min)
            max_val = parse_math_number(raw_max)

            # Split in case multiple variables share a bound, like "a_i, T"
            variables = [v.strip() for v in raw_vars.split(",") if v.strip()]
            for var in variables:
                extracted_bounds.append({
                    "variable": var,
                    "min_value": min_val,
                    "max_value": max_val,
                    "source_math": span  # Provenance (receipt!)
                })
            continue

        # Pattern B: Single upper bound like "n <= 200000"
        single_match = re.search(r'([a-zA-Z0-9_,\s]+)\s*<=\s*([-\d\.\*\^\\a-zA-Z,]+)', clean)
        if single_match:
            raw_vars, raw_max = single_match.groups()
            max_val = parse_math_number(raw_max)
            variables = [v.strip() for v in raw_vars.split(",") if v.strip()]
            for var in variables:
                extracted_bounds.append({
                    "variable": var,
                    "min_value": None,
                    "max_value": max_val,
                    "source_math": span
                })

    return extracted_bounds


# ==========================================================
# JOB 3: Custom NER (Extract Data Structures & Clue Phrases)
# ==========================================================
def extract_cp_entities(simplified_task_text):
    """
    Scans the simplified problem statement to recognize CP Data Structures
    and Algorithmic Condition Cues.
    """
    lower_text = simplified_task_text.lower()
    entities = []

    # Dictionary of Custom NER Categories and their trigger phrases
    ner_dictionary = {
        "DATA_STRUCTURE": ["array", "string", "tree", "graph", "matrix", "grid", "permutation"],
        "CONDITION_CUE": [
            "two distinct elements", "pair", "contiguous subarray", "subsequence",
            "sorted", "non-decreasing", "connected", "undirected", "directed"
        ],
        "OPERATION_CUE": ["add up to", "sum", "xor", "minimum", "maximum", "shortest path"]
    }

    for category, phrases in ner_dictionary.items():
        for phrase in phrases:
            if phrase in lower_text:
                entities.append({
                    "entity_text": phrase,
                    "entity_type": category
                })

    return entities


# ==========================================================
# JOB 4: Estimate Feasible Time Complexities from 'n'
# ==========================================================
def estimate_complexity_feasibility(bounds_list, time_limit_sec=1.0):
    """
    Finds the primary input size variable (like n, m, or q) and estimates
    which time complexities are safe vs likely to TLE under 10^8 ops/sec.
    """
    ops_budget = int(1e8 * time_limit_sec) # 100,000,000 ops per second

    # Look for standard size variables ('n', 'm', 'q', 'k') first
    primary_bound = None
    for b in bounds_list:
        if b["variable"] in ("n", "N", "m", "M", "q", "Q") and b["max_value"] is not None:
            primary_bound = b
            break

    if not primary_bound:
        return {"status": "No primary size variable (n, m, q) found."}

    n = primary_bound["max_value"]

    # Calculate approximate operations for each complexity class
    complexities = [
        ("O(n)", n),
        ("O(n log n)", int(n * math.log2(max(2, n)))),
        ("O(n sqrt(n))", int(n * math.isqrt(n))),
        ("O(n^2)", n ** 2),
        ("O(n^3)", n ** 3),
        ("O(2^n)", 2 ** n if n <= 30 else float('inf')) # Avoid huge number overflow
    ]

    evaluation = []
    for name, est_ops in complexities:
        if est_ops <= ops_budget // 5:
            verdict = "SAFE (Should easily pass)"
        elif est_ops <= ops_budget:
            verdict = "BORDERLINE (Depends on constant factor)"
        else:
            verdict = "LIKELY TLE (Too slow for this bound)"

        evaluation.append({
            "complexity": name,
            "estimated_operations": est_ops if est_ops != float('inf') else "> 10^9",
            "verdict": verdict
        })

    return {
        "primary_variable": primary_bound["variable"],
        "max_size": n,
        "assumed_ops_budget": ops_budget,
        "complexity_table": evaluation
    }


# ==========================================================
# PUTTING MODULE 1 + MODULE 2 TOGETHER
# ==========================================================
def run_module_2_baseline(problem_text):
    # First get Module 1's output
    m1_result = run_module_1_baseline(problem_text)

    # Collect all math spans from CONSTRAINT and INPUT_SPEC sentences
    all_math_spans = []
    for sent in m1_result["tagged_sentences"]:
        if sent["role"] in ("CONSTRAINT", "INPUT_SPEC"):
            all_math_spans.extend(sent["extracted_math_spans"])

    # Run Module 2 jobs
    extracted_bounds = extract_constraints_from_math(all_math_spans)
    cp_entities = extract_cp_entities(m1_result["simplified_task"])
    complexity_guide = estimate_complexity_feasibility(extracted_bounds)

    return {
        "simplified_task": m1_result["simplified_task"],
        "extracted_entities": cp_entities,
        "parsed_constraints": extracted_bounds,
        "complexity_guidance": complexity_guide
    }


# --- TEST MODULE 1 + MODULE 2 TOGETHER ---
if __name__ == "__main__":
    sample_problem = (
        "Alice and Bob are visiting a magical park with many attractions. "
        "You are given an array of $n$ integers $a_1, a_2, ..., a_n$ representing ticket prices. "
        "Determine whether two distinct elements add up to a target $T$. "
        "It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i, T <= 10^9$."
    )

    output = run_module_2_baseline(sample_problem)
    print(json.dumps(output, indent=2))