import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--size", type=int, default=10, help="Deep Sea grid size N")
    parser.add_argument("--prior_reward", type=float, default=1.0, help="Prior optimistic reward mean")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45]
    episodes = args.episodes
    size = args.size
    prior_reward = args.prior_reward
    os.makedirs("results_deepsea", exist_ok=True)
    r_tag = f"_r{int(prior_reward)}" if prior_reward != 1.0 else ""
    out_combined = f"results_deepsea/presampling_4seeds_N{size}{r_tag}_{episodes}ep.json"

    print("=" * 80)
    print(f"PARALLEL CPU BENCHMARK: 4 Seeds Across 4 Independent CPU Processes (N={size}, r={prior_reward})")
    print(f"Seeds: {seeds} | Episodes: {episodes} per seed | Total Runs: {len(seeds) * 2}")
    print("=" * 80, flush=True)

    t0 = time.time()
    processes = []
    
    # Launch 4 independent OS processes, each pinned to 1 CPU core via PyTorch num_threads=1
    for seed in seeds:
        cmd = [sys.executable, "run_seed_worker.py", "--seed", str(seed), "--episodes", str(episodes), "--size", str(size), "--prior_reward", str(prior_reward)]
        print(f"Spawning CPU worker process for Seed {seed} (N={size}, r={prior_reward})...")
        p = subprocess.Popen(cmd)
        processes.append((seed, p))

    print(f"\nAll {len(processes)} CPU worker processes running in parallel! Waiting for completion...", flush=True)

    # Wait for all processes to finish
    for seed, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Warning: Worker for seed {seed} exited with code {p.returncode}")

    total_wall_clock = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL PARALLEL CPU WORKERS FINISHED! Total Wall-Clock Time: {total_wall_clock:.2f}s")
    print("=" * 80)

    # Gather results from each worker
    all_results = []
    for seed in seeds:
        seed_file = f"results_deepsea/seed_{seed}_N{size}{r_tag}.json"
        if os.path.exists(seed_file):
            with open(seed_file, "r") as f:
                all_results.extend(json.load(f))

    with open(out_combined, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary Table
    print(f"\nBENCHMARK RESULTS TABLE (Deep Sea {size}x{size}, {episodes:,} Episodes):")
    print(f"{'Method':<32} | {'Seed':<5} | {'Solved':<7} | {'T_first':<8} | {'T_learn':<8} | {'Time (s)':<9} | {'Speed (eps/s)':<13}")
    print("-" * 80)
    for r in all_results:
        t_first = str(r['first_discovery']) if r['first_discovery'] is not None else "N/A"
        t_learn = str(r['learn_time']) if r['learn_time'] is not None else "N/A"
        print(f"{r['agent']:<32} | {r['seed']:<5} | {str(r['solved']):<7} | {t_first:<8} | {t_learn:<8} | {r['elapsed_seconds']:<9.2f} | {r['episodes_per_sec']:<13.1f}")
    print("=" * 80)

    # Aggregate Statistics
    without_runs = [r for r in all_results if r['mode'] == 'Without Pre-sampling']
    with_runs = [r for r in all_results if r['mode'] == 'With Pre-sampling']

    spd_without = [r['episodes_per_sec'] for r in without_runs]
    spd_with = [r['episodes_per_sec'] for r in with_runs]
    time_without = [r['elapsed_seconds'] for r in without_runs]
    time_with = [r['elapsed_seconds'] for r in with_runs]

    print(f"\nAVERAGE SPEED COMPARISON (Across 4 Seeds, {episodes:,} Episodes Each):")
    print(f"  Without Pre-sampling: {np.mean(spd_without):.1f} +/- {np.std(spd_without):.1f} eps/s (Mean Time: {np.mean(time_without):.2f}s)")
    print(f"  With Pre-sampling:    {np.mean(spd_with):.1f} +/- {np.std(spd_with):.1f} eps/s (Mean Time: {np.mean(time_with):.2f}s)")
    speed_up = (np.mean(spd_with) - np.mean(spd_without)) / np.mean(spd_without) * 100
    print(f"  Throughput Advantage: {speed_up:+.2f}%")
    print("=" * 80)

    # Call plotting scripts
    print("\nGenerating multi-seed publication figures...")
    subprocess.run([sys.executable, "plot_4seeds_comparison.py", "--results_file", out_combined], check=True)
    subprocess.run([sys.executable, "plot_regret_analysis.py", "--results_file", out_combined], check=True)
    subprocess.run([sys.executable, "plot_focused_regret.py", "--results_file", out_combined], check=True)
    print("Done!")

if __name__ == "__main__":
    main()
