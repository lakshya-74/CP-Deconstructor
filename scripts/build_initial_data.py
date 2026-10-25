import json
import os

# 1. TRAINING PROBLEMS (Used to train CRF and Pretrained Transformer Classifier)
train_data = [
    {
        "problem_id": "TRAIN_01",
        "source": "codeforces_style",
        "sentences": [
            {"text": "Monocarp is playing a video game where he collects magical coins.", "role": "STORY"},
            {"text": "You are given an array $a$ consisting of $n$ integers.", "role": "INPUT_SPEC"},
            {"text": "Find the maximum sum of a contiguous subarray of length $k$.", "role": "TASK"},
            {"text": "It is guaranteed that $1 <= k <= n <= 10^5$ and $-10^9 <= a_i <= 10^9$.", "role": "CONSTRAINT"},
            {"text": "For example, if $a = [1, 4, 2, 10]$ and $k = 2$, the answer is $12$.", "role": "EXAMPLE"}
        ]
    },
    {
        "problem_id": "TRAIN_02",
        "source": "codechef_style",
        "sentences": [
            {"text": "In the year $3000$, Chef bought $5$ robots to clean his restaurant.", "role": "STORY"},
            {"text": "Chef always wanted to find the fastest way to clean all tables.", "role": "STORY"},
            {"text": "The restaurant is represented by a grid of $n$ rows and $m$ columns.", "role": "INPUT_SPEC"},
            {"text": "Your goal is to calculate the minimum time required to visit all dirty cells.", "role": "TASK"},
            {"text": "The sum of $n \\cdot m$ over all test cases does not exceed $200000$.", "role": "CONSTRAINT"}
        ]
    },
    {
        "problem_id": "TRAIN_03",
        "source": "codeforces_style",
        "sentences": [
            {"text": "Polycarp loves strings and decided to invent a new puzzle.", "role": "STORY"},
            {"text": "Given a string $s$ of length $n$ containing lowercase English letters.", "role": "INPUT_SPEC"},
            {"text": "Return the length of the longest palindrome that can be formed.", "role": "TASK"},
            {"text": "Here $1 <= n <= 2 \\cdot 10^5$.", "role": "CONSTRAINT"},
            {"text": "In the first test case, the longest palindrome has length $7$.", "role": "EXAMPLE"}
        ]
    },
    {
        "problem_id": "TRAIN_04",
        "source": "cses_style",
        "sentences": [
            {"text": "There are $n$ islands connected by $m$ bridges in a magical archipelago.", "role": "INPUT_SPEC"},
            {"text": "Each bridge has a durability weight $w_i$.", "role": "INPUT_SPEC"},
            {"text": "Determine whether it is possible to travel from island $1$ to island $n$.", "role": "TASK"},
            {"text": "It is guaranteed that $2 <= n <= 10^5$ and $1 <= w_i <= 10^9$.", "role": "CONSTRAINT"}
        ]
    },
    {
        "problem_id": "TRAIN_05",
        "source": "codeforces_style",
        "sentences": [
            {"text": "Alice and Bob are playing a game with $2$ piles of stones.", "role": "STORY"},
            {"text": "Initially, you are given $n$ piles where the $i$-th pile has $c_i$ stones.", "role": "INPUT_SPEC"},
            {"text": "Count the number of winning first moves for Alice.", "role": "TASK"},
            {"text": "The value of $n$ does not exceed $50000$.", "role": "CONSTRAINT"},
            {"text": "For example, when $c = [3, 4, 5]$, Alice has $1$ winning move.", "role": "EXAMPLE"}
        ]
    },
    {
        "problem_id": "TRAIN_06",
        "source": "codeforces_style",
        "sentences": [
            {"text": "During his vacation, Vasya visited $10$ ancient castles.", "role": "STORY"},
            {"text": "He wanted to find the oldest artifact in the museum.", "role": "STORY"},
            {"text": "An array $p$ of $n$ integers denotes the age of each artifact.", "role": "INPUT_SPEC"},
            {"text": "Your goal is to find the minimum cost to buy at least $k$ artifacts.", "role": "TASK"},
            {"text": "Constraints: $1 <= k <= n <= 200000$ and $1 <= p_i <= 10^6$.", "role": "CONSTRAINT"}
        ]
    }
]

# 2. GOLD TEST SET (Includes TEST_04 with unseen story vocabulary!)
gold_test_data = [
    {
        "problem_id": "TEST_01",
        "source": "benchmark",
        "sentences": [
            {"text": "Alice and Bob are visiting a magical park.", "role": "STORY"},
            {"text": "You are given an array of $n$ integers $a_1, a_2, ..., a_n$.", "role": "INPUT_SPEC"},
            {"text": "Determine whether two distinct elements add up to $T$.", "role": "TASK"},
            {"text": "It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i <= 10^9$.", "role": "CONSTRAINT"},
            {"text": "For example, if $a = [2, 7, 11]$ and $T = 9$, the answer is YES.", "role": "EXAMPLE"}
        ]
    },
    {
        "problem_id": "TEST_02",
        "source": "benchmark",
        "sentences": [
            {"text": "In the year $2050$, Byteasar bought $3$ magical swords for his kingdom.", "role": "STORY"},
            {"text": "The kingdom consists of $n$ cities connected by $m$ bidirectional roads.", "role": "INPUT_SPEC"},
            {"text": "Your goal is to make all cities reachable from city $1$ using the minimum total cost.", "role": "TASK"},
            {"text": "Here $1 \\le n, m \\le 10^5$.", "role": "CONSTRAINT"}
        ]
    },
    {
        "problem_id": "TEST_03",
        "source": "benchmark",
        "sentences": [
            {"text": "Bob always wanted to find the secret treasure hidden in the mountains.", "role": "STORY"},
            {"text": "An array $b$ of length $n$ contains the heights of the mountains.", "role": "INPUT_SPEC"},
            {"text": "Return the length of the longest contiguous subarray with sum equal to $k$.", "role": "TASK"},
            {"text": "The sum of $n$ over all test cases does not exceed $200000$.", "role": "CONSTRAINT"},
            {"text": "In the first test case, the subarray $[1, 3]$ has sum $5$.", "role": "EXAMPLE"}
        ]
    },
    {
        "problem_id": "TEST_04",
        "source": "unseen_vocab_benchmark",
        "sentences": [
            {"text": "Captain Zara piloted her starship across $7$ distant galaxies.", "role": "STORY"},
            {"text": "She hoped to find a new home planet for her crew.", "role": "STORY"},
            {"text": "A sequence $d_1, d_2, \\dots, d_n$ represents the fuel required for each jump.", "role": "INPUT_SPEC"},
            {"text": "Compute the minimum initial fuel needed to complete the journey.", "role": "TASK"},
            {"text": "Note that $1 \\le n \\le 10^5$ and $1 \\le d_i \\le 10^9$.", "role": "CONSTRAINT"}
        ]
    }
]

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    with open("data/train_sentences.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open("data/gold_test_set.json", "w", encoding="utf-8") as f:
        json.dump(gold_test_data, f, indent=2)
    print("Updated data/train_sentences.json and data/gold_test_set.json (with TEST_04)!")