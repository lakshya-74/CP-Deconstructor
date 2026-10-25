import re
import json

def protect_latex_and_split_sentences(problem_text):
    """
    Takes a raw CP problem statement, protects the LaTeX math ($...$),
    splits the text into clean sentences, and restores the math.
    """
    # Dictionary to safely store our math formulas while we split sentences
    math_vault = {}
    counter = 0

    # This helper function runs every time Python finds a $...$ formula
    def store_math(match):
        nonlocal counter
        placeholder = f"__MATH_{counter}__"
        math_vault[placeholder] = match.group(0) # Save the original math like "$1 \le n \le 10^5$"
        counter += 1
        return placeholder

    # 1. Replace all $$...$$ and $...$ math blocks with __MATH_0__, __MATH_1__, etc.
    protected_text = re.sub(r'(\$\$[\s\S]*?\$\$|\$[^\$]+?\$)', store_math, problem_text)

    # 2. Split the text into sentences whenever we see a period/question mark/exclamation mark
    #    followed by a space or newline.
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected_text.strip())

    # 3. Put the original math back into each sentence and record which math belonged where
    final_sentences = []
    for index, sent in enumerate(raw_sentences):
        restored_sent = sent
        math_in_this_sentence = []

        # Check if any __MATH_X__ placeholder is inside this sentence
        for placeholder, original_math in math_vault.items():
            if placeholder in restored_sent:
                restored_sent = restored_sent.replace(placeholder, original_math)
                math_in_this_sentence.append(original_math)

        final_sentences.append({
            "sentence_id": index,
            "protected_view": sent,          # How the model sees it without messy math
            "original_sentence": restored_sent, # The real sentence with math restored
            "extracted_math_spans": math_in_this_sentence # Already helps Module 2!
        })

    return final_sentences


# --- LET'S TEST IT ON A SAMPLE CP PROBLEM ---
if __name__ == "__main__":
    sample_problem = (
        "Alice and Bob are visiting a magical park with many attractions. "
        "You are given an array of $n$ integers $a_1, a_2, ..., a_n$ representing ticket prices. "
        "Determine whether two distinct elements add up to a target $T$. "
        "It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i, T <= 10^9$."
    )

    result = protect_latex_and_split_sentences(sample_problem)
    print(json.dumps(result, indent=2))