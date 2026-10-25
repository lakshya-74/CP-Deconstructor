import re
import json
from old_basic.step1_preprocessor import protect_latex_and_split_sentences

def classify_sentence_role(sentence_dict, total_sentences):
    """
    Takes one sentence dictionary from Step 1 and assigns a Discourse Role label
    along with the 'provenance' (the exact clue that triggered the label).
    """
    orig = sentence_dict["original_sentence"]
    lower_sent = orig.lower()
    math_spans = sentence_dict["extracted_math_spans"]
    sent_id = sentence_dict["sentence_id"]

    # --- RULE 1: Check for CONSTRAINT ---
    # Clue: Contains inequality symbols (<=, >=, \le, \ge, <, >) inside math OR constraint keywords
    inequality_pattern = r'(<=|>=|\\le|\\ge|\\leq|\\geq|<|>|10\^)'
    has_inequality_math = any(re.search(inequality_pattern, m) for m in math_spans)
    
    if has_inequality_math or "guaranteed that" in lower_sent or "constraints:" in lower_sent:
        return {
            "role": "CONSTRAINT",
            "evidence": f"Contains inequality math bounds: {math_spans}"
        }

    # --- RULE 2: Check for EXAMPLE ---
    # Clue: Mentions sample test cases or examples
    example_keywords = ["for example", "in the first test case", "sample input", "explanation:"]
    for kw in example_keywords:
        if kw in lower_sent:
            return {
                "role": "EXAMPLE",
                "evidence": f"Matched example keyword: '{kw}'"
            }

    # --- RULE 3: Check for TASK (The actual goal) ---
    # Clue: Action verbs asking the student to compute/find/output something, or a question mark
    task_keywords = [
        "determine whether", "find the", "calculate the", "compute the",
        "output the", "print the", "what is the", "count the number",
        "minimum number of", "maximum possible"
    ]
    for kw in task_keywords:
        if kw in lower_sent:
            return {
                "role": "TASK",
                "evidence": f"Matched task trigger phrase: '{kw}'"
            }
    if orig.endswith("?"):
        return {
            "role": "TASK",
            "evidence": "Sentence ends with a direct question ('?')"
        }

    # --- RULE 4: Check for INPUT_SPEC (Defines the data structures/variables) ---
    # Clue: Phrases like "you are given", "array of", "consisting of", OR has math variables
    input_keywords = [
        "you are given", "given an", "given a", "consists of",
        "first line contains", "represented by", "denotes the"
    ]
    for kw in input_keywords:
        if kw in lower_sent:
            return {
                "role": "INPUT_SPEC",
                "evidence": f"Matched input specification phrase: '{kw}'"
            }

    # If a sentence has math variables (like $n$ or $a_i$) and wasn't a constraint or task,
    # it is usually defining part of the problem setup.
    if len(math_spans) > 0:
        return {
            "role": "INPUT_SPEC",
            "evidence": f"Contains mathematical variables {math_spans} defining problem objects"
        }

    # --- RULE 5: Default to STORY (Narrative fluff) ---
    # Clue: No math variables, no task/input keywords (especially common at the start of problems)
    return {
        "role": "STORY",
        "evidence": f"Sentence {sent_id + 1}/{total_sentences} has no math variables or task keywords (narrative context)"
    }


def run_module_1_baseline(problem_text):
    """
    Runs the complete Module 1 Baseline:
    1. Splits sentences while protecting LaTeX math (Step 1)
    2. Labels each sentence with its discourse role + evidence (Step 2)
    3. Strips out the STORY sentences to create a clean Simplified Task
    """
    sentences = protect_latex_and_split_sentences(problem_text)
    total = len(sentences)

    for sent in sentences:
        prediction = classify_sentence_role(sent, total)
        sent["role"] = prediction["role"]
        sent["evidence"] = prediction["evidence"]

    # Build the Simplified Task by keeping only INPUT_SPEC and TASK sentences
    essential_sentences = [
        s["original_sentence"] for s in sentences 
        if s["role"] in ("INPUT_SPEC", "TASK")
    ]
    simplified_task = " ".join(essential_sentences)

    return {
        "simplified_task": simplified_task,
        "tagged_sentences": sentences
    }


# --- TEST MODULE 1 ON OUR SAMPLE PROBLEM ---
if __name__ == "__main__":
    sample_problem = (
        "Alice and Bob are visiting a magical park with many attractions. "
        "You are given an array of $n$ integers $a_1, a_2, ..., a_n$ representing ticket prices. "
        "Determine whether two distinct elements add up to a target $T$. "
        "It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i, T <= 10^9$."
    )

    module1_output = run_module_1_baseline(sample_problem)
    print(json.dumps(module1_output, indent=2))