"""Concurrent multiprocessing pool runner for unified DP-DQN benchmarks."""

import os
import sys
import argparse
import multiprocessing as mp
import subprocess
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BIN = sys.executable


def run_job(task):
    env_name, size, algo, seed, alpha, episodes, out_dir, warmstart_steps = task
    script = os.path.join(CURRENT_DIR, "run_single.py")
    log_file = os.path.join(out_dir, f"{env_name}_{algo}_s{seed}.log")
    cmd = [
        PYTHON_BIN, "-u", script,
        "--env", env_name,
        "--size", str(size),
        "--algo", algo,
        "--seed", str(seed),
        "--alpha", str(alpha),
        "--episodes", str(episodes),
        "--out_dir", out_dir
    ]
    if warmstart_steps is not None:
        cmd.extend(["--warmstart_steps", str(warmstart_steps)])

    t0 = time.time()
    with open(log_file, "w") as lf:
        res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"[SUCCESS] {env_name} (N={size}) | {algo} (seed {seed}) finished in {elapsed:.1f}s")
        return (algo, seed, True, elapsed)
    else:
        print(f"[FAILED] {env_name} (N={size}) | {algo} (seed {seed}) failed with return code {res.returncode}. See {log_file}")
        return (algo, seed, False, elapsed)


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Multiprocessing Pool")
    parser.add_argument("--env", type=str, default="cartpole_swingup",
                        help="Environment name")
    parser.add_argument("--size", type=int, default=20,
                        help="Size parameter (for deep_sea N)")
    parser.add_argument("--algos", nargs="+",
                        default=["dp_dqn_haar", "dp_dqn_non_haar", "boot_dqn", "bdqn", "dp_dqn_alpha_small", "vanilla_dqn"],
                        help="Algorithms to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(42, 52)),
                        help="Random seeds to evaluate")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="Dirichlet Process concentration parameter")
    parser.add_argument("--episodes", type=int, default=2500,
                        help="Episodes per run")
    parser.add_argument("--warmstart_steps", type=int, default=None,
                        help="Episodic warmstart steps")
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()),
                        help="Number of concurrent worker processes")
    parser.add_argument("--out_dir", type=str, default="./results_unified",
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for algo in args.algos:
        for seed in args.seeds:
            tasks.append((args.env, args.size, algo, seed, args.alpha, args.episodes, args.out_dir, args.warmstart_steps))


    total = len(tasks)
    print("=" * 60)
    print(f" Launching Unified Benchmark Pool")
    print(f" Environment : {args.env}")
    print(f" Algorithms  : {args.algos}")
    print(f" Seeds       : {args.seeds} ({len(args.seeds)} seeds)")
    print(f" Total Runs  : {total}")
    print(f" Concurrency : {args.workers} parallel workers")
    print(f" Output Dir  : {args.out_dir}")
    print("=" * 60)

    t_start = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(run_job, tasks)

    total_time = time.time() - t_start
    n_success = sum(1 for r in results if r[2])
    print("\n" + "=" * 60)
    print(f" Benchmark Complete in {total_time/60:.2f} minutes!")
    print(f" Success Rate: {n_success}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
