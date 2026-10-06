"""Concurrent Pool Launcher for 60 Runs (6 Algorithms × 10 Seeds) on Cart-Pole Swing-Up.

Methods:
1. `dp_dqn_haar`: DP-DQN Haar (alpha=3.0)
2. `dp_dqn_non_haar`: DP-DQN Non-Haar (alpha=3.0)
3. `boot_dqn`: BootDQN + Randomized Prior (Osband et al., 2018)
4. `bdqn`: Bayesian Deep Q-Networks (Azizzadenesheli et al., 2018)
5. `dp_dqn_alpha_small`: DP-DQN alpha = 1e-10 (Zero Epistemic Variance Limit)
6. `vanilla_dqn`: Vanilla DQN (Epsilon-greedy annealing)

Seeds: 42 to 51 (10 seeds each) = 60 total runs.
"""

import os
import sys
import argparse
import multiprocessing as mp
import subprocess
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = sys.executable

ALGOS = [
    "dp_dqn_haar",
    "dp_dqn_non_haar",
    "boot_dqn",
    "bdqn",
    "dp_dqn_alpha_small",
    "vanilla_dqn"
]
SEEDS = list(range(42, 52))  # 10 seeds: 42, 43, ..., 51


def run_single_job(args_tuple):
    algo, seed, episodes, out_dir = args_tuple
    script = os.path.join(REPO_ROOT, "src", "run_cartpole_full_benchmark.py")
    cmd = [
        PYTHON_BIN, "-u", script,
        "--algo", algo,
        "--seed", str(seed),
        "--episodes", str(episodes),
        "--out_dir", out_dir
    ]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"[SUCCESS] {algo} (seed {seed}) finished in {elapsed:.1f}s")
        return (algo, seed, True, elapsed, "")
    else:
        print(f"[FAILED] {algo} (seed {seed}) failed with return code {res.returncode}")
        print(f"Error:\n{res.stderr[-500:]}")
        return (algo, seed, False, elapsed, res.stderr)


def main():
    parser = argparse.ArgumentParser(description="Launch 60-run benchmark pool")
    parser.add_argument("--workers", type=int, default=min(60, mp.cpu_count()),
                        help="Number of concurrent worker processes")
    parser.add_argument("--episodes", type=int, default=2500,
                        help="Number of episodes per run")
    parser.add_argument("--out_dir", type=str, default="./results_rl/cartpole_60runs",
                        help="Output directory for results and checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for algo in ALGOS:
        for seed in SEEDS:
            tasks.append((algo, seed, args.episodes, args.out_dir))

    total = len(tasks)
    print(f"==================================================")
    print(f"Launching Cart-Pole Swing-Up 60-Run Benchmark")
    print(f"Total tasks: {total} ({len(ALGOS)} algorithms × {len(SEEDS)} seeds)")
    print(f"Concurrency: {args.workers} parallel workers")
    print(f"Output directory: {args.out_dir}")
    print(f"Episodes per run: {args.episodes}")
    print(f"==================================================")

    t_start = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(run_single_job, tasks)

    total_time = time.time() - t_start
    n_success = sum(1 for r in results if r[2])
    print(f"\n==================================================")
    print(f"Benchmark finished in {total_time/60:.2f} minutes!")
    print(f"Successful runs: {n_success}/{total}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
