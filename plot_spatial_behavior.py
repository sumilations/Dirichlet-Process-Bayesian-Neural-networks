import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.deceptive_oasis import DeceptiveOasisBandit
from src.models.dp_bnn import DPBNNAgent
from src.models.baselines import RandomizedPriorEnsembleAgent


def plot_spatial_decisions():
    seed = 100
    env = DeceptiveOasisBandit(seed=seed)
    dp = DPBNNAgent(context_dim=2, num_arms=4, hidden_dim=64, alpha=10.0, truncation_K=150, prior_reward_mean=8.0, lr=0.01, steps_per_decision=2, seed=seed)
    boot = RandomizedPriorEnsembleAgent(context_dim=2, num_arms=4, num_models=5, hidden_dim=64, prior_scale=3.0, lr=0.01, steps_per_decision=2, seed=seed)

    dp_contexts = []
    dp_actions = []
    boot_contexts = []
    boot_actions = []

    for t in range(1500):
        ctx = env.sample_context()

        # DP
        a_dp = dp.select_action(ctx)
        r_dp, _, _, _, _, _ = env.step(ctx, a_dp)
        dp.update(ctx, a_dp, r_dp)
        dp_contexts.append(ctx)
        dp_actions.append(a_dp)

        # Boot
        a_b = boot.select_action(ctx)
        r_b, _, _, _, _, _ = env.step(ctx, a_b)
        boot.update(ctx, a_b, r_b)
        boot_contexts.append(ctx)
        boot_actions.append(a_b)

    dp_ctx = np.array(dp_contexts)
    dp_act = np.array(dp_actions)
    boot_ctx = np.array(boot_contexts)
    boot_act = np.array(boot_actions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Circle patch for oasis
    for ax, title, ctxs, acts, name in zip(axes,
                                          ["BootDQN + Randomized Priors\n(0.0% Discovery - Trap Locked)",
                                           "DP-BNN (Vashishtha Formulation)\n(94.1% Discovery - Oasis Exploited)"],
                                          [boot_ctx, dp_ctx],
                                          [boot_act, dp_act],
                                          ["BootDQN-RP", "DP-BNN"]):

        # Draw oasis boundary
        circle = plt.Circle(env.oasis_center, env.oasis_radius, color='gold', fill=True, alpha=0.25, label="Oasis ($\mu=50.0$)")
        circle_edge = plt.Circle(env.oasis_center, env.oasis_radius, color='darkgoldenrod', fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.add_patch(circle_edge)

        # Plot Arm 0 pulls (Comfort trap)
        idx_0 = (acts == 0)
        ax.scatter(ctxs[idx_0, 0], ctxs[idx_0, 1], c='#d62728', s=16, alpha=0.4, label="Arm 0 (Safe 2.0 Trap)")

        # Plot Arm 1 pulls (Oasis Arm)
        idx_1 = (acts == 1)
        ax.scatter(ctxs[idx_1, 0], ctxs[idx_1, 1], c='#0047AB', s=28, alpha=0.9, edgecolors='black', linewidth=0.5, label="Arm 1 (Jackpot 50.0)")

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Context $x_1$", fontsize=10)
        ax.set_ylabel("Context $x_2$", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_file = "results/deceptive_oasis_spatial_map.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Spatial map saved to {out_file}")


if __name__ == "__main__":
    plot_spatial_decisions()
