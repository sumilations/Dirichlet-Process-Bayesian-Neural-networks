"""Concurrent subprocess pool runner for Cart-Pole Swing-Up Pure TS Benchmark.

Tests:
1. Pure TS (sample_once_per_episode = True): Single DP draw per episode.
2. Multi-Sample (sample_once_per_episode = False): Morning construct baseline.
Both with Gaussian State Base Measure (s ~ N(0, I)) and Optimistic Reward Prior (r ~ N(0.5, 0.5)).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import subprocess
import sys
import time
import concurrent.futures

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BIN = sys.executable
RUN_SINGLE = os.path.join(CURRENT_DIR, "unified_dp_dqn", "run_single.py")


def run_cartpole_job(task):
    alpha, seed, pure_ts, episodes, out_dir = task
    mode_str = "pure_ts" if pure_ts else "multi_sample"
    tag = f"gaussian_{mode_str}_a{alpha}"
    log_file = os.path.join(out_dir, f"cartpole_{mode_str}_a{alpha}_s{seed}.log")
    json_file = os.path.join(out_dir, f"cartpole_swingup_dp_dqn_deepsea_construct_{tag}_s{seed}.json")

    # Check if already completed
    if os.path.exists(json_file):
        print(f"[CACHED] CartPole {mode_str} a={alpha} s={seed} already completed -> {json_file}", flush=True)
        return (mode_str, alpha, seed, True, 0.0)

    cmd = [
        PYTHON_BIN, "-u", RUN_SINGLE,
        "--env", "cartpole_swingup",
        "--algo", "dp_dqn_deepsea_construct",
        "--base_measure", "gaussian",
        "--alpha", str(alpha),
        "--seed", str(seed),
        "--episodes", str(episodes),
        "--tag", tag,
        "--out_dir", out_dir
    ]
    if pure_ts:
        cmd.append("--sample_once_per_episode")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"[SUCCESS] CartPole {mode_str.upper()} a={alpha} s={seed} finished in {elapsed:.1f}s", flush=True)
        return (mode_str, alpha, seed, True, elapsed)
    else:
        print(f"[FAILED] CartPole {mode_str.upper()} a={alpha} s={seed} failed (code {proc.returncode}). See {log_file}", flush=True)
        return (mode_str, alpha, seed, False, elapsed)


def main():
    parser = argparse.ArgumentParser(description="Cart-Pole Swing-Up Pure TS Benchmark Pool")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker subprocesses")
    parser.add_argument("--episodes", type=int, default=2500, help="Episodes per run")
    parser.add_argument("--alphas", nargs="+", type=float, default=[3.0, 15.0], help="Alpha values")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--include_multi_sample", action="store_true", default=True, help="Also run multi-sample baseline")
    parser.add_argument("--out_dir", type=str, default="./results_cartpole_pure_ts", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    # 1. Pure TS tasks
    for a in args.alphas:
        for s in args.seeds:
            tasks.append((a, s, True, args.episodes, args.out_dir))

    # 2. Multi-sample tasks (for direct ablation comparison)
    if args.include_multi_sample:
        for a in args.alphas:
            for s in args.seeds:
                tasks.append((a, s, False, args.episodes, args.out_dir))

    print("=" * 65)
    print(" Cart-Pole Swing-Up Pure TS vs Multi-Sample Benchmark Pool")
    print(f" Total tasks : {len(tasks)}")
    print(f" Workers     : {args.workers}")
    print(f" Alphas      : {args.alphas}")
    print(f" Seeds       : {args.seeds}")
    print(f" Episodes    : {args.episodes}")
    print(f" Output Dir  : {args.out_dir}")
    print("=" * 65, flush=True)

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_cartpole_job, task): task for task in tasks}
        for fut in concurrent.futures.as_completed(futures):
            mode, a, s, success, elap = fut.result()
            print(f"--> [COMPLETE] {mode.upper()} a={a} s={s} | Success={success} | Elapsed={elap:.1f}s", flush=True)

    print(f"All {len(tasks)} tasks finished in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
