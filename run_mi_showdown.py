"""
Mutual Information Benchmark Showdown:
Benchmarks DP-BNN, DP-MINE, MINE, NWJ, InfoNCE, and KSG across:
1. Correlated Gaussian (variable correlation rho, d=2)
2. High-Dimensional Scaling (d in {1, 2, 5, 10, 20}, rho=0.6)
3. Non-linear Relationships (Cubic, Sinusoid, Circular Ring)

Saves detailed JSON metrics to results_mi/mi_results.json.
"""

import os
import json
import time
import argparse
import numpy as np
import torch

from src.envs.mi_benchmark_datasets import (
    gaussian_true_mi,
    generate_correlated_gaussian,
    generate_cubic,
    generate_sinusoid,
    generate_circular
)
from src.models.dp_bnn_mi import (
    KSG_Estimator,
    MINE_Estimator,
    NWJ_Estimator,
    InfoNCE_Estimator,
    DPMINE_Estimator,
    DPBNN_MIEstimator
)


def train_and_eval_estimator(est_name: str, dim_x: int, dim_y: int,
                            x_tr: torch.Tensor, y_tr: torch.Tensor,
                            x_te: torch.Tensor, y_te: torch.Tensor,
                            n_steps: int = 350, batch_size: int = 128) -> float:
    """Trains a given estimator and returns the test evaluation."""
    N = x_tr.size(0)

    if est_name == "KSG":
        ksg = KSG_Estimator(k=3)
        return ksg.estimate(x_te.cpu().numpy(), y_te.cpu().numpy())

    elif est_name == "MINE":
        model = MINE_Estimator(dim_x, dim_y, hidden_dim=64, lr=2e-3)
        for _ in range(n_steps):
            idx = torch.randint(0, N, (batch_size,))
            model.step(x_tr[idx], y_tr[idx])
        return model.estimate(x_te, y_te)

    elif est_name == "NWJ":
        model = NWJ_Estimator(dim_x, dim_y, hidden_dim=64, lr=2e-3)
        for _ in range(n_steps):
            idx = torch.randint(0, N, (batch_size,))
            model.step(x_tr[idx], y_tr[idx])
        return model.estimate(x_te, y_te)

    elif est_name == "InfoNCE":
        model = InfoNCE_Estimator(dim_x, dim_y, hidden_dim=64, lr=2e-3)
        for _ in range(n_steps):
            idx = torch.randint(0, N, (batch_size,))
            model.step(x_tr[idx], y_tr[idx])
        return model.estimate(x_te, y_te)

    elif est_name == "DP-MINE":
        model = DPMINE_Estimator(dim_x, dim_y, num_particles=5, hidden_dim=64, alpha=1.0, lr=2e-3)
        for _ in range(n_steps):
            idx = torch.randint(0, N, (batch_size,))
            model.step(x_tr[idx], y_tr[idx])
        return model.estimate(x_te, y_te)

    elif est_name == "DP-BNN (Predictive MI)":
        # For non-linear, single-directional X->Y is well-defined; for Gaussian, symmetric works well
        model = DPBNN_MIEstimator(dim_x, dim_y, num_particles=8, hidden_dim=64, alpha=1.0, lr=2e-3, symmetric=False)
        for _ in range(n_steps):
            idx = torch.randint(0, N, (batch_size,))
            model.train_step(x_tr[idx], y_tr[idx])
        return model.estimate(x_te, y_te)

    else:
        raise ValueError(f"Unknown estimator {est_name}")


def run_benchmark(n_seeds: int = 3, out_dir: str = "results_mi"):
    os.makedirs(out_dir, exist_ok=True)

    estimators = ["DP-BNN (Predictive MI)", "DP-MINE", "MINE", "NWJ", "InfoNCE", "KSG"]
    seeds = [42 + i * 100 for i in range(n_seeds)]

    results = {
        "gaussian_correlation_sweep": {},
        "dimension_scaling_sweep": {},
        "nonlinear_benchmarks": {}
    }

    start_time = time.time()
    print("=" * 70)
    print(f"STARTING MUTUAL INFORMATION BENCHMARK SHOWDOWN ({n_seeds} seeds)")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Correlated Gaussian Sweep (dim=2, rho in [0.1, 0.99])
    # --------------------------------------------------------------------------
    print("\n--- Benchmark 1: Correlated Gaussian Sweep (dim=2) ---")
    rhos = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
    dim = 2

    for rho in rhos:
        true_mi = gaussian_true_mi(dim, rho)
        rho_key = f"rho_{rho:.2f}"
        print(f"Evaluating rho={rho:.2f} (True MI = {true_mi:.3f} nats)...")

        results["gaussian_correlation_sweep"][rho_key] = {
            "rho": rho,
            "true_mi": true_mi,
            "estimators": {}
        }

        for est_name in estimators:
            seed_estimates = []
            for s in seeds:
                X_tr, Y_tr, _ = generate_correlated_gaussian(1000, dim=dim, rho=rho, seed=s)
                X_te, Y_te, _ = generate_correlated_gaussian(1000, dim=dim, rho=rho, seed=s + 999)

                x_tr_t = torch.from_numpy(X_tr)
                y_tr_t = torch.from_numpy(Y_tr)
                x_te_t = torch.from_numpy(X_te)
                y_te_t = torch.from_numpy(Y_te)

                est = train_and_eval_estimator(est_name, dim, dim, x_tr_t, y_tr_t, x_te_t, y_te_t)
                seed_estimates.append(est)

            mean_est = float(np.mean(seed_estimates))
            std_est = float(np.std(seed_estimates) / np.sqrt(n_seeds))
            results["gaussian_correlation_sweep"][rho_key]["estimators"][est_name] = {
                "mean": mean_est,
                "stderr": std_est,
                "mae": float(abs(mean_est - true_mi)),
                "raw": seed_estimates
            }
            print(f"  {est_name:24s}: {mean_est:.3f} +/- {std_est:.3f} (Error: {abs(mean_est - true_mi):.3f})")

    # --------------------------------------------------------------------------
    # 2. High-Dimensional Scaling Sweep (rho=0.6, dim in [1, 2, 5, 10, 20])
    # --------------------------------------------------------------------------
    print("\n--- Benchmark 2: High-Dimensional Scaling (rho=0.6) ---")
    dims = [1, 2, 5, 10, 20]
    fixed_rho = 0.6

    for d in dims:
        true_mi = gaussian_true_mi(d, fixed_rho)
        d_key = f"dim_{d}"
        print(f"Evaluating dim={d} (True MI = {true_mi:.3f} nats)...")

        results["dimension_scaling_sweep"][d_key] = {
            "dim": d,
            "true_mi": true_mi,
            "estimators": {}
        }

        for est_name in estimators:
            seed_estimates = []
            for s in seeds:
                X_tr, Y_tr, _ = generate_correlated_gaussian(1000, dim=d, rho=fixed_rho, seed=s)
                X_te, Y_te, _ = generate_correlated_gaussian(1000, dim=d, rho=fixed_rho, seed=s + 999)

                x_tr_t = torch.from_numpy(X_tr)
                y_tr_t = torch.from_numpy(Y_tr)
                x_te_t = torch.from_numpy(X_te)
                y_te_t = torch.from_numpy(Y_te)

                est = train_and_eval_estimator(est_name, d, d, x_tr_t, y_tr_t, x_te_t, y_te_t)
                seed_estimates.append(est)

            mean_est = float(np.mean(seed_estimates))
            std_est = float(np.std(seed_estimates) / np.sqrt(n_seeds))
            results["dimension_scaling_sweep"][d_key]["estimators"][est_name] = {
                "mean": mean_est,
                "stderr": std_est,
                "mae": float(abs(mean_est - true_mi)),
                "raw": seed_estimates
            }
            print(f"  {est_name:24s}: {mean_est:.3f} +/- {std_est:.3f} (Error: {abs(mean_est - true_mi):.3f})")

    # --------------------------------------------------------------------------
    # 3. Non-linear Benchmarks: Cubic, Sinusoid, Circular Ring
    # --------------------------------------------------------------------------
    print("\n--- Benchmark 3: Non-Linear Benchmarks ---")
    nl_tasks = [
        ("Cubic (Y=X^3+eps)", generate_cubic, 1, 1),
        ("Sinusoid (Y=sin(2X)+eps)", generate_sinusoid, 1, 1),
        ("Circular Ring (rho=0)", generate_circular, 1, 1)
    ]

    for task_name, gen_func, dx, dy in nl_tasks:
        _, _, true_mi = gen_func(1000, seed=42)
        print(f"Evaluating {task_name} (Approx True MI = {true_mi:.3f} nats)...")

        results["nonlinear_benchmarks"][task_name] = {
            "true_mi": true_mi,
            "estimators": {}
        }

        for est_name in estimators:
            seed_estimates = []
            for s in seeds:
                X_tr, Y_tr, _ = gen_func(1000, seed=s)
                X_te, Y_te, _ = gen_func(1000, seed=s + 999)

                x_tr_t = torch.from_numpy(X_tr)
                y_tr_t = torch.from_numpy(Y_tr)
                x_te_t = torch.from_numpy(X_te)
                y_te_t = torch.from_numpy(Y_te)

                est = train_and_eval_estimator(est_name, dx, dy, x_tr_t, y_tr_t, x_te_t, y_te_t)
                seed_estimates.append(est)

            mean_est = float(np.mean(seed_estimates))
            std_est = float(np.std(seed_estimates) / np.sqrt(n_seeds))
            results["nonlinear_benchmarks"][task_name]["estimators"][est_name] = {
                "mean": mean_est,
                "stderr": std_est,
                "mae": float(abs(mean_est - true_mi)),
                "raw": seed_estimates
            }
            print(f"  {est_name:24s}: {mean_est:.3f} +/- {std_est:.3f}")

    total_time = time.time() - start_time
    results["total_time_sec"] = total_time
    out_file = os.path.join(out_dir, "mi_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE in {total_time:.1f}s. Results written to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="results_mi")
    args = parser.parse_args()
    run_benchmark(n_seeds=args.seeds, out_dir=args.out_dir)
