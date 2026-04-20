import requests
import time
import json
import psutil
import os
import statistics

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"

# Test prompts of varying lengths
TEST_PROMPTS = {
    "short": [
        {"role": "user", "content": "Hi, what products do you have?"}
    ],
    "medium": [
        {"role": "user", "content": "Hi, what products do you have?"},
        {"role": "assistant", "content": "Welcome to FreshMart! We have Fruits, Vegetables, Dairy, Bakery, Beverages, and Snacks. What would you like to browse?"},
        {"role": "user", "content": "Add 2 apples and 1 milk to my cart. How much is that?"}
    ],
    "long": [
        {"role": "user", "content": "Hi, what products do you have?"},
        {"role": "assistant", "content": "Welcome to FreshMart! We have Fruits, Vegetables, Dairy, Bakery, Beverages, and Snacks. What would you like to browse?"},
        {"role": "user", "content": "Add 2 apples and 1 milk."},
        {"role": "assistant", "content": "Added 2 Apples ($1.50 each) and 1 Milk ($2.00). Cart total: $5.00. Anything else?"},
        {"role": "user", "content": "Do you have any discounts on fruits?"},
        {"role": "assistant", "content": "Yes! We currently have a 10% discount on all fruits this week. Your apples will now cost $2.70 instead of $3.00. Updated total: $4.70."},
        {"role": "user", "content": "Great, also add 1 bread. What are your delivery slots and fees?"}
    ]
}

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Shabo, the FreshMart Virtual Assistant. "
        "Help customers browse products, manage their cart, and answer store policy questions. "
        "Be friendly, helpful, and concise. Stay within the grocery domain only."
    )
}


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_prompt(label, messages, runs=5):
    print(f"\n{'='*50}")
    print(f"Testing: {label.upper()} context ({len(messages)} turns)")
    print(f"{'='*50}")

    ttft_list = []
    total_time_list = []
    tokens_per_sec_list = []
    token_counts = []

    for run in range(1, runs + 1):
        payload = {
            "model": MODEL,
            "messages": [SYSTEM_PROMPT] + messages,
            "stream": True
        }

        start_time = time.time()
        first_token_time = None
        full_response = ""
        token_count = 0

        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            full_response += content
                            token_count += 1
                        if chunk.get("done"):
                            break

            end_time = time.time()
            ttft = (first_token_time - start_time) if first_token_time else None
            total_time = end_time - start_time
            tps = token_count / total_time if total_time > 0 else 0

            ttft_list.append(ttft)
            total_time_list.append(total_time)
            tokens_per_sec_list.append(tps)
            token_counts.append(token_count)

            print(f"  Run {run}: TTFT={ttft:.3f}s | Total={total_time:.3f}s | Tokens={token_count} | TPS={tps:.1f}")

        except Exception as e:
            print(f"  Run {run}: ERROR - {e}")

    if ttft_list:
        print(f"\n  --- Averages over {runs} runs ---")
        print(f"  Avg TTFT        : {statistics.mean(ttft_list):.3f}s")
        print(f"  Avg Total Time  : {statistics.mean(total_time_list):.3f}s")
        print(f"  Avg Tokens/sec  : {statistics.mean(tokens_per_sec_list):.1f}")
        print(f"  Avg Token Count : {statistics.mean(token_counts):.0f}")

    return {
        "label": label,
        "runs": runs,
        "avg_ttft": statistics.mean(ttft_list) if ttft_list else None,
        "avg_total_time": statistics.mean(total_time_list) if total_time_list else None,
        "avg_tps": statistics.mean(tokens_per_sec_list) if tokens_per_sec_list else None,
        "avg_tokens": statistics.mean(token_counts) if token_counts else None,
    }


def main():
    print("FreshMart Chatbot - Ollama Qwen 1.5B Benchmark")
    print(f"Model : {MODEL}")
    print(f"Ollama: {OLLAMA_URL}")

    # Check Ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Available models: {models}")
        if not any(MODEL.split(":")[0] in m for m in models):
            print(f"\nWARNING: {MODEL} not found. Run: ollama pull {MODEL}")
    except Exception as e:
        print(f"\nERROR: Cannot connect to Ollama. Is it running? ({e})")
        return

    mem_before = get_memory_usage_mb()
    print(f"\nMemory before benchmarking: {mem_before:.1f} MB")

    results = []
    for label, messages in TEST_PROMPTS.items():
        result = benchmark_prompt(label, messages, runs=5)
        results.append(result)

    mem_after = get_memory_usage_mb()
    print(f"\nMemory after benchmarking : {mem_after:.1f} MB")

    # Summary table
    print("\n" + "="*65)
    print("SUMMARY TABLE")
    print("="*65)
    print(f"{'Context':<10} {'TTFT (s)':<12} {'Total (s)':<12} {'TPS':<10} {'Tokens':<8}")
    print("-"*65)
    for r in results:
        print(f"{r['label']:<10} {r['avg_ttft']:<12.3f} {r['avg_total_time']:<12.3f} {r['avg_tps']:<10.1f} {r['avg_tokens']:<8.0f}")
    print("="*65)
    print("\nBenchmark complete. Copy the summary table into your README.md.")


if __name__ == "__main__":
    main()