import re
import json

# Common abbreviations whose periods should NEVER split a sentence
ABBREVIATIONS = ["e.g.", "i.e.", "mr.", "mrs.", "dr.", "vs.", "etc.", "fig."]


def has_algebraic_variable(math_spans):
    """
    Checks if any math span in the list contains an algebraic variable letter (like n, a_i, k)
    rather than just plain numbers (like $2050$ or $3$).
    """
    for span in math_spans:
        # Strip outer math delimiters ($, $$, \(, \), \[, \])
        clean = re.sub(r'^(\$\$|\$|\\\(|\\\[)|(\$\$|\$|\\\)|\\\])$', '', span.strip())
        # Remove LaTeX commands like \le, \ge, \dots, \times, \cdot, \sum, \in
        clean = re.sub(r'\\[a-zA-Z]+', '', clean)
        # If any English letter remains, it is a real mathematical variable
        if re.search(r'[a-zA-Z]', clean):
            return True
    return False


def preprocess_and_split_sentences(problem_text):
    """
    Bulletproof Module 1 Preprocessor:
    1. Normalizes whitespace and protects all 4 types of LaTeX math: $$...$$, $...$, \(...\), \[...\]
    2. Protects common abbreviations (e.g., i.e., Mr., etc.) so their periods don't break sentences
    3. Splits text into clean sentences and restores all original math and abbreviations
    """
    if not problem_text or not problem_text.strip():
        return []

    math_vault = {}
    math_counter = 0

    def store_math(match):
        nonlocal math_counter
        placeholder = f"__MATH_{math_counter}__"
        # Normalize \( ... \) and \[ ... \] to standard $ ... $ so downstream modules stay consistent
        raw_math = match.group(0)
        if raw_math.startswith(r"\(") and raw_math.endswith(r"\)"):
            raw_math = "$" + raw_math[2:-2].strip() + "$"
        elif raw_math.startswith(r"\[") and raw_math.endswith(r"\]"):
            raw_math = "$$" + raw_math[2:-2].strip() + "$$"

        math_vault[placeholder] = raw_math
        math_counter += 1
        return placeholder

    # Step 1: Protect $$...$$, $...$, \(...\), and \[...\]
    latex_pattern = r'(\$\$[\s\S]*?\$\$|\$[^\$]+?\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])'
    protected_text = re.sub(latex_pattern, store_math, problem_text)

    # Step 2: Protect periods inside abbreviations (like "e.g." -> "e<DOT>g<DOT>")
    for abbr in ABBREVIATIONS:
        safe_abbr = abbr.replace(".", "<DOT>")
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        protected_text = pattern.sub(safe_abbr, protected_text)

    # Step 3: Split sentences on '.', '!', or '?' followed by whitespace or newline
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected_text.strip())

    # Step 4: Restore abbreviations and LaTeX math into each sentence
    final_sentences = []
    for idx, sent in enumerate(raw_sentences):
        if not sent.strip():
            continue

        # Restore abbreviation dots
        clean_protected = sent.replace("<DOT>", ".")
        restored_sent = clean_protected
        math_in_sentence = []

        # Sort keys by length descending so __MATH_10__ is replaced before __MATH_1__
        for placeholder in sorted(math_vault.keys(), key=len, reverse=True):
            original_math = math_vault[placeholder]
            if placeholder in restored_sent:
                restored_sent = restored_sent.replace(placeholder, original_math)
                math_in_sentence.append(original_math)

        final_sentences.append({
            "sentence_id": idx,
            "protected_view": clean_protected,
            "original_sentence": restored_sent,
            "extracted_math_spans": math_in_sentence,
            "has_real_variable": has_algebraic_variable(math_in_sentence)
        })

    return final_sentences


# --- BUILT-IN EDGE CASE UNIT TEST ---
if __name__ == "__main__":
    tricky_text = (
        "Mr. Bob visited 10 cities (e.g., city A and city B) in the year $2050$. "
        "You are given an array $a_1, a_2, \\dots, a_n$ of size \\(1 \\le n \\le 2 \\cdot 10^5\\). "
        "Find the optimal value, i.e., the minimum possible maximum sum!"
    )

    output = preprocess_and_split_sentences(tricky_text)
    print(f"Total sentences split: {len(output)} (Expected: 3)\n")
    print(json.dumps(output, indent=2))