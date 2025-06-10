import subprocess
import json
import math
import time

INITIAL_STACK = 1000
ROUNDS_PER_BASELINE = 5
SCRIPT_NAME = "start_game.py"  # change this if your filename is different

def run_match(baseline_index):
    results = []
    wins = 0
    start_time = time.time()
    for game_num in range(ROUNDS_PER_BASELINE):
        print(f"  → Game {game_num + 1}")
        try:
            output = subprocess.check_output(
                ["python", SCRIPT_NAME, "-b", str(baseline_index)],
                universal_newlines=True
            )

            action_count = output.count("[ACTION]")
            timeout_count = output.count("[TIMEOUT]")
            if timeout_count > 0:
                print(f"    \033[91mTimeout count: {timeout_count} / {action_count}\033[0m")
            output = output[output.find('{\n    "rule":'):]

            lines = output.strip().split('\n')
            json_start = next(i for i, line in enumerate(lines) if line.strip().startswith('{'))
            json_str = '\n'.join(lines[json_start:])
            result = json.loads(json_str)
            results.append(result)

            if nottako_won(result):
                wins += 1
                if wins == 3:
                    print("  → nottako wins 3 games — skipping remainder.")
                    break

        except subprocess.CalledProcessError as e:
            print(f"Error running game {game_num + 1} vs baseline {baseline_index}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    end_time = time.time()
    print(f"    Runtime: {end_time - start_time:.2f}s")
    return results

def get_nottako_result(result):
    for p in result["players"]:
        if p["name"] == "nottako":
            return p["stack"]
    raise ValueError("nottako not found")

def nottako_won(result):
    stacks = [p["stack"] for p in result["players"]]
    return max(stacks) == get_nottako_result(result)

def grade_baseline(results):
    wins = sum(nottako_won(r) for r in results)
    if wins >= 3:
        return 5.0

    # fallback scoring
    stack_data = []
    for r in results:
        stack = get_nottako_result(r)
        won = nottako_won(r)
        stack_data.append((stack, won))

    top2 = sorted(stack_data, reverse=True, key=lambda x: x[0])[:2]
    score = 0.0
    for stack, won in top2:
        if won:
            score += 1.5
        else:
            ratio = stack / INITIAL_STACK
            if ratio >= 0.5:
                score += round(ratio, 1)
    return min(score, 3.0)

def main():
    total_score = 0.0
    for i in range(1, 8):  # Only baseline 1 through 7
        print(f"\nRunning baseline {i}...")
        results = run_match(i)
        score = grade_baseline(results)
        total_score += score
        print(f"    Score: {score:.1f}")
    print(f"\nTotal Score: {total_score:.1f}")

if __name__ == "__main__":
    main()
