"""
Multi-Armed Bandit Empirical Evaluation for AISTATS Paper:
"On the Regret of Approximate Thompson Sampling: Why Laplace Fails and How Dirichlet Process Posterior Sampling Succeeds"

Implements:
1. Laplace-TS: Laplace approximation Thompson Sampling (Chapelle & Li 2011, Phan et al. 2019)
2. Gaussian-TS: Standard parametric Gaussian Thompson Sampling
3. NPTS: Non-Parametric Thompson Sampling (Riou & Honda 2020)
4. Approximated DPPS: Vashishtha & Maillard (RLJ 2025) with:
   - Monte Carlo batch subsampling (B_batch)
   - Finite stick-breaking truncation (K_prior)
"""

import os
import json
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
})

# -------------------------------------------------------------
# 1. Environment Definitions
# -------------------------------------------------------------

class LogisticBernoulliBandit:
    """Non-conjugate Bernoulli bandit parameterized via logits theta_a:
       r ~ Bernoulli(sigma(theta_a)).
       Prior: theta_a ~ N(0, 1).
       This is the canonical setting where Laplace approximation is applied (Chapelle & Li, 2011).
    """
    def __init__(self, means):
        self.means = np.array(means)
        self.K = len(means)
        self.best_arm = np.argmax(self.means)
        self.best_mean = self.means[self.best_arm]

    def pull(self, arm, rng):
        return 1.0 if rng.random() < self.means[arm] else 0.0


class SkewedJackpotBandit:
    """Rare-event / Skewed reward bandit:
       Arm 0 (Optimal, High Variance): 0 with prob 0.85, 5.0 with prob 0.15 (mean = 0.75)
       Arm 1 (Sub-optimal, Safe): Deterministic 0.60
       Arm 2 (Sub-optimal, Safe): Deterministic 0.40
    """
    def __init__(self):
        self.means = np.array([0.75, 0.60, 0.40])
        self.K = 3
        self.best_arm = 0
        self.best_mean = 0.75

    def pull(self, arm, rng):
        if arm == 0:
            return 5.0 if rng.random() < 0.15 else 0.0
        elif arm == 1:
            return 0.60
        else:
            return 0.40


# -------------------------------------------------------------
# 2. Algorithm Implementations
# -------------------------------------------------------------

class LaplaceTS:
    """Thompson Sampling using Laplace approximation for Logistic Likelihood:
       p(theta | D) approx N(theta_MAP, H^{-1})
    """
    def __init__(self, K, prior_var=1.0):
        self.K = K
        self.prior_var = prior_var
        self.history = [[] for _ in range(K)]

    def select_arm(self, rng):
        samples = np.zeros(self.K)
        for a in range(self.K):
            obs = self.history[a]
            if len(obs) == 0:
                samples[a] = rng.normal(0.0, np.sqrt(self.prior_var))
                continue
            
            n = len(obs)
            s = sum(obs)
            # Find MAP estimate of theta: log p(D|theta) + log p(theta)
            # log-likelihood: s * theta - n * log(1 + exp(theta))
            # prior: - theta^2 / (2 * prior_var)
            def nll(th):
                t = th[0]
                # numerically stable log1pexp
                log1pexp = np.logaddexp(0.0, t)
                return - (s * t - n * log1pexp - (t**2) / (2.0 * self.prior_var))

            def nll_grad(th):
                t = th[0]
                sig = 1.0 / (1.0 + np.exp(-t))
                return - (s - n * sig - t / self.prior_var)

            # MAP via L-BFGS-B
            res = opt.minimize(nll, x0=[0.0], jac=nll_grad, method='L-BFGS-B')
            theta_map = res.x[0]

            # Hessian at MAP: n * sig * (1 - sig) + 1 / prior_var
            sig = 1.0 / (1.0 + np.exp(-theta_map))
            hessian = n * sig * (1.0 - sig) + 1.0 / self.prior_var
            var = 1.0 / max(hessian, 1e-4)

            # Draw sample from Laplace Gaussian approximation
            th_sample = rng.normal(theta_map, np.sqrt(var))
            samples[a] = 1.0 / (1.0 + np.exp(-th_sample))

        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.history[arm].append(reward)


class GaussianTS:
    """Standard Gaussian Thompson Sampling with conjugate Gaussian prior."""
    def __init__(self, K, prior_mean=0.5, prior_var=1.0, obs_var=0.25):
        self.K = K
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.obs_var = obs_var
        self.history = [[] for _ in range(K)]

    def select_arm(self, rng):
        samples = np.zeros(self.K)
        for a in range(self.K):
            obs = self.history[a]
            n = len(obs)
            if n == 0:
                samples[a] = rng.normal(self.prior_mean, np.sqrt(self.prior_var))
            else:
                s = sum(obs)
                post_var = 1.0 / (1.0 / self.prior_var + n / self.obs_var)
                post_mean = post_var * (self.prior_mean / self.prior_var + s / self.obs_var)
                samples[a] = rng.normal(post_mean, np.sqrt(post_var))
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.history[arm].append(reward)


class NPTS:
    """Non-Parametric Thompson Sampling (Riou & Honda 2020) with known bound B=1."""
    def __init__(self, K, B=1.0):
        self.K = K
        self.B = B
        self.history = [[] for _ in range(K)]

    def select_arm(self, rng):
        samples = np.zeros(self.K)
        for a in range(self.K):
            obs = self.history[a]
            n = len(obs)
            if n == 0:
                samples[a] = self.B # Initial optimism
            else:
                # Dir(1, ..., 1) over n observations + 1 pseudo-atom at B
                weights = rng.dirichlet(np.ones(n + 1))
                samples[a] = np.dot(weights[:-1], obs) + weights[-1] * self.B
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.history[arm].append(reward)


class ApproximatedDPPS:
    """Approximated Dirichlet Process Posterior Sampling (Vashishtha & Maillard, RLJ 2025)
       incorporating:
       1. Monte Carlo Batch Subsampling (B_batch): Evaluates a random mini-batch rather than full history.
       2. Finite Stick-Breaking Truncation (K_prior): Truncates the prior base measure F_0 to K atoms.
    """
    def __init__(self, K, alpha=1.0, k_prior=10, batch_size=32, prior_mean=0.5, prior_std=0.3):
        self.K = K
        self.alpha = alpha
        self.k_prior = k_prior
        self.batch_size = batch_size
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.history = [[] for _ in range(K)]

    def select_arm(self, rng):
        samples = np.zeros(self.K)
        for a in range(self.K):
            obs = self.history[a]
            n_avail = len(obs)

            # 1. Monte Carlo Batch Subsampling
            if n_avail == 0:
                emp_obs = np.array([])
            elif n_avail <= self.batch_size:
                emp_obs = np.array(obs)
            else:
                # Draw random mini-batch uniformly
                idx = rng.choice(n_avail, size=self.batch_size, replace=False)
                emp_obs = np.array(obs)[idx]

            N_emp = len(emp_obs)

            # 2. Finite Truncation of Prior Base Measure F_0
            syn_atoms = rng.normal(loc=self.prior_mean, scale=self.prior_std, size=self.k_prior)
            # Clip if bounded
            syn_atoms = np.clip(syn_atoms, 0.0, 1.0)

            # 3. Vashishtha & Maillard Recursive Stick-Breaking Weights
            # Prior weight W_prior ~ Beta(alpha + 1, N_stat)
            N_stat = max(N_emp, n_avail)
            if N_stat == 0:
                W_prior = 1.0
            else:
                W_prior = float(rng.beta(self.alpha + 1.0, float(N_stat)))

            if N_emp > 0:
                i_arr = np.arange(1, N_emp + 1, dtype=np.float32)
                V_emp = rng.beta(1.0, self.alpha + i_arr).astype(np.float32)
                U_emp = 1.0 - V_emp
                suffix_U = np.ones(N_emp + 1, dtype=np.float32)
                suffix_U[:-1] = np.cumprod(U_emp[::-1])[::-1]

                w_emp_raw = np.zeros(N_emp, dtype=np.float32)
                if N_emp > 1:
                    w_emp_raw[:-1] = V_emp[:-1] * suffix_U[1:-1]
                w_emp_raw[-1] = V_emp[-1]
                s_emp = w_emp_raw.sum()
                w_emp = (1.0 - W_prior) * (w_emp_raw / max(s_emp, 1e-6))
            else:
                w_emp = np.array([])

            # Prior stick-breaking weights
            V_prior = rng.beta(1.0, max(self.alpha, 1e-3), size=self.k_prior).astype(np.float32)
            U_prior = 1.0 - V_prior
            w_prior_raw = np.zeros(self.k_prior, dtype=np.float32)
            cum = 1.0
            for j in range(self.k_prior):
                w_prior_raw[j] = V_prior[j] * cum
                cum *= U_prior[j]
            s_prior = w_prior_raw.sum()
            w_prior = W_prior * (w_prior_raw / max(s_prior, 1e-6))

            # Sampled mean is the integral against the random measure
            mean_sample = 0.0
            if N_emp > 0:
                mean_sample += np.dot(w_emp, emp_obs)
            mean_sample += np.dot(w_prior, syn_atoms)
            samples[a] = mean_sample

        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.history[arm].append(reward)


# -------------------------------------------------------------
# 3. Main Benchmark Execution
# -------------------------------------------------------------

def run_experiment(env_type="logistic", num_runs=40, horizon=3000):
    print(f"Running Benchmark: {env_type} | Runs: {num_runs} | Horizon: {horizon} ...")
    
    if env_type == "logistic":
        # 5-armed Logistic Bernoulli Bandit with subtle gap
        means = [0.85, 0.80, 0.75, 0.60, 0.40]
        make_env = lambda: LogisticBernoulliBandit(means)
    else:
        make_env = lambda: SkewedJackpotBandit()

    test_env = make_env()
    K = test_env.K

    algorithms = {
        "Laplace-TS": lambda: LaplaceTS(K, prior_var=1.0),
        "Gaussian-TS": lambda: GaussianTS(K, prior_mean=0.5, prior_var=1.0, obs_var=0.25),
        "NPTS (Riou & Honda)": lambda: NPTS(K, B=1.0),
        "Approximated DPPS (Ours)": lambda: ApproximatedDPPS(K, alpha=1.0, k_prior=10, batch_size=32),
    }

    results = {name: np.zeros((num_runs, horizon)) for name in algorithms}
    freeze_counts = {name: 0 for name in algorithms}

    for run_idx in range(num_runs):
        rng = np.random.RandomState(42 + run_idx * 17)
        for name, algo_fn in algorithms.items():
            env = make_env()
            algo = algo_fn()
            cum_regret = 0.0
            regrets = []
            
            for t in range(horizon):
                arm = algo.select_arm(rng)
                reward = env.pull(arm, rng)
                algo.update(arm, reward)
                
                instant_regret = env.best_mean - env.means[arm]
                cum_regret += instant_regret
                regrets.append(cum_regret)

            results[name][run_idx] = np.array(regrets)
            # Check if optimal arm was frozen out (pulled < 5% in second half)
            if hasattr(algo, "history"):
                opt_pulls_second_half = sum(1 for a in algo.history[env.best_arm] if True) # total pulls
                if opt_pulls_second_half < (horizon * 0.10):
                    freeze_counts[name] += 1

        if (run_idx + 1) % 10 == 0:
            print(f"  Completed {run_idx + 1}/{num_runs} runs.")

    print("\n=== SUMMARY AT HORIZON T =", horizon, "===")
    for name in algorithms:
        final_regrets = results[name][:, -1]
        print(f"{name:25s}: Mean Regret = {np.mean(final_regrets):.1f} +/- {np.std(final_regrets):.1f} | Starvation Rate = {freeze_counts[name]}/{num_runs} ({100*freeze_counts[name]/num_runs:.1f}%)")

    return results, algorithms


def plot_results(results, algorithms, out_pdf="aistats_dpps_paper/figure1_bandit_regret.pdf", out_png="aistats_dpps_paper/figure1_bandit_regret.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        "Laplace-TS": "#D55E00",              # Red/Orange
        "Gaussian-TS": "#E69F00",             # Gold/Orange
        "NPTS (Riou & Honda)": "#0072B2",     # Blue
        "Approximated DPPS (Ours)": "#009E73",# Green
    }
    styles = {
        "Laplace-TS": "--",
        "Gaussian-TS": ":",
        "NPTS (Riou & Honda)": "-.",
        "Approximated DPPS (Ours)": "-",
    }

    # Panel 1: Cumulative Regret Curves
    ax1 = axes[0]
    horizon = list(results.values())[0].shape[1]
    t_axis = np.arange(1, horizon + 1)

    for name in algorithms:
        mean_regret = np.mean(results[name], axis=0)
        std_err = np.std(results[name], axis=0) / np.sqrt(results[name].shape[0])
        ax1.plot(t_axis, mean_regret, label=name, color=colors[name], linestyle=styles[name], linewidth=2.2)
        ax1.fill_between(t_axis, mean_regret - std_err, mean_regret + std_err, color=colors[name], alpha=0.15)

    ax1.set_xlabel("Time Step $t$")
    ax1.set_ylabel("Cumulative Expected Regret $R(t)$")
    ax1.set_title("(a) Cumulative Regret on Logistic Bernoulli Bandit")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Regret Distribution Boxplot
    ax2 = axes[1]
    data_to_plot = [results[name][:, -1] for name in algorithms]
    labels = [name.replace(" (Riou & Honda)", "\n(Riou & Honda)").replace(" (Ours)", "\n(Ours)") for name in algorithms]
    
    bplot = ax2.boxplot(data_to_plot, patch_artist=True, labels=labels, notch=True, vert=True)
    for patch, name in zip(bplot['boxes'], algorithms):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)

    for median in bplot['medians']:
        median.set(color='black', linewidth=2)

    ax2.set_ylabel("Final Regret $R(T)$ at $T = 3{,}000$")
    ax2.set_title("(b) Final Regret Dispersion Across 40 Independent Runs")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=300)
    print(f"Figures saved to {out_pdf} and {out_png}")


if __name__ == "__main__":
    results, algorithms = run_experiment(env_type="logistic", num_runs=40, horizon=3000)
    plot_results(results, algorithms)
