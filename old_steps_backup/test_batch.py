import json
from old_basic.step4_pattern_and_hints import run_full_mvp_pipeline

# A tiny test dataset of 3 distinct competitive programming problems
sample_dataset = [
    {
        "id": "CF_1",
        "title": "Two Sum Style Problem",
        "text": (
            "You are given an array of $n$ integers $a_1, a_2, ..., a_n$. "
            "Find two distinct indices $i$ and $j$ such that $a_i + a_j = T$. "
            "It is guaranteed that $1 <= n <= 200000$ and $1 <= a_i, T <= 10^9$."
        )
    },
    {
        "id": "CF_2",
        "title": "Tree / Graph Problem",
        "text": (
            "You are given a connected undirected tree consisting of $n$ vertices and $n-1$ edges. "
            "Calculate the shortest path distance from vertex 1 to all other vertices. "
            "The constraints are $1 <= n <= 10^5$ with a time limit of 2.0 seconds."
        )
    },
    {
        "id": "CF_3",
        "title": "Min-Max / Binary Search Problem",
        "text": (
            "There are $n$ tasks, and the $i$-th task takes $a_i$ hours. "
            "You want to find the minimum possible maximum workload among $k$ workers. "
            "It is guaranteed that $1 <= n <= 50000$ and $1 <= a_i <= 10^9$."
        )
    }
]

def run_batch_evaluation():
    print(f"=== RUNNING BATCH TEST ON {len(sample_dataset)} PROBLEMS ===\n")
    
    for item in sample_dataset:
        print(f"--- Processing [{item['id']}]: {item['title']} ---")
        
        # Run the full MVP pipeline we built in Step 4
        result = run_full_mvp_pipeline(item["text"])
        
        # Print key results cleanly
        print(f"  > Simplified Task: {result['1_simplified_task']}")
        print(f"  > Primary Variable & Bound: {result['4_complexity_guidance'].get('primary_variable')} <= {result['4_complexity_guidance'].get('max_size')}")
        
        top_patterns = [p['pattern'] for p in result['5_ranked_algorithmic_patterns']]
        print(f"  > Top Predicted Patterns: {top_patterns}")
        
        first_hint = result['6_progressive_hints'][0]['hint']
        print(f"  > Stage 1 Hint Preview: {first_hint[:90]}...\n")

if __name__ == "__main__":
    run_batch_evaluation()