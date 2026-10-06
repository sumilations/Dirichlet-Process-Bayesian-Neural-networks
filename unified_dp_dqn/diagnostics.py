"""Comprehensive Diagnostic Suite for DP-DQN on Cart-Pole Swing-Up.

Visualizes:
1. (theta, theta_dot) Value Landscape V(s) and Policy Quiver Field.
2. Replay Buffer Phase Space Coverage (Density of stored states).
3. Warmstart Perturbation Shift Delta Q across episodes.
4. Posterior Effective Sample Size (ESS) and W_prior evolution.
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from unified_dp_dqn import DPDQNAgent, DPDQNConfig, make_env


def run_diagnostic(
    episodes: int = 250,
    seed: int = 42,
    hidden_dim: int = 64,
    use_layer_norm: bool = False,
    sampler_type: str = "dirichlet",
    alpha: float = 3.0,
    out_img_path: str = "dp_dqn_diagnostic_dashboard.png"
):
    print(f"=== Running DP-DQN Diagnostic Suite ({episodes} episodes, seed {seed}, hidden {hidden_dim}) ===")
    
    env = make_env("cartpole_swingup", seed=seed)
    cfg = DPDQNConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=use_layer_norm,
        alpha=alpha,
        base_measure_type="haar",
        haar_angle=True,
        sampler_type=sampler_type,
        w_min=0.05,
        seed=seed,
        buffer_capacity=1000000,
        warmstart_steps=2,
        dp_online_sgd=False,
        tau=0.05
    )
    agent = DPDQNAgent(cfg)

    # Fixed evaluation probe states across (theta, theta_dot)
    th_eval = np.linspace(-np.pi, np.pi, 25)
    th_dot_eval = np.linspace(-6.0, 6.0, 25)
    TH_ev, THD_ev = np.meshgrid(th_eval, th_dot_eval)
    eval_states = []
    for t, td in zip(TH_ev.ravel(), THD_ev.ravel()):
        eval_states.append([np.cos(t), np.sin(t), td, 0.0, 0.0, 0.5])
    eval_states_t = torch.tensor(eval_states, dtype=torch.float32)

    delta_q_history = []
    w_prior_history = []
    ess_history = []
    pos_buffer_ratio = []
    returns = []

    t0 = time.time()
    for ep in range(1, episodes + 1):
        # Measure Q-values BEFORE warmstart
        with torch.no_grad():
            agent.q_net.eval()
            q_before = agent.q_net(eval_states_t).clone()

        # Episode reset (runs warmstart on q_net)
        agent.reset_episode()

        # Measure Q-values AFTER warmstart
        with torch.no_grad():
            agent.q_net.eval()
            q_after = agent.q_net(eval_states_t)
            delta_q = torch.abs(q_after - q_before).mean().item()
            delta_q_history.append(delta_q)

        # Run episode
        state = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            ep_ret += reward
            state = next_state

        returns.append(ep_ret)

        # Record buffer stats
        buf_size = len(agent.replay)
        pos_ratio = agent.replay.num_positive_rewards / max(1, buf_size)
        pos_buffer_ratio.append(pos_ratio)

        # Sample one batch to record posterior weights
        if buf_size >= agent.config.batch_size:
            _, _, _, _, _, q_w = agent.sampler.sample(agent.replay)
            w_np = q_w.cpu().numpy()
            ess = 1.0 / np.sum(w_np**2)
            ess_history.append(ess)
            # W_prior is the weight mass of the prior atoms (last K atoms)
            K_prior = max(8, int(agent.config.vm_prior_multiplier * agent.config.alpha))
            w_prior_history.append(float(np.sum(w_np[-K_prior:])))
        else:
            ess_history.append(0.0)
            w_prior_history.append(0.0)

        if ep % 25 == 0 or ep == episodes:
            print(f"Ep {ep:3d}/{episodes} | Ret: {np.mean(returns[-25:]):6.2f} | "
                  f"DeltaQ: {delta_q:5.2f} | ESS: {ess_history[-1]:4.1f} | "
                  f"PosBuffer: {pos_ratio*100:4.1f}% | Time: {time.time() - t0:4.1f}s", flush=True)

    print("Generating 4-panel diagnostic dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- PANEL 1: (theta, theta_dot) Value Landscape & Policy Quiver ---
    ax1 = axes[0, 0]
    grid_th = np.linspace(-np.pi, np.pi, 60)
    grid_thd = np.linspace(-8.0, 8.0, 60)
    G_TH, G_THD = np.meshgrid(grid_th, grid_thd)
    grid_states = []
    for t, td in zip(G_TH.ravel(), G_THD.ravel()):
        grid_states.append([np.cos(t), np.sin(t), td, 0.0, 0.0, 0.5])
    grid_states_t = torch.tensor(grid_states, dtype=torch.float32)

    with torch.no_grad():
        agent.q_net.eval()
        q_grid = agent.q_net(grid_states_t)
        v_grid = q_grid.max(dim=1)[0].numpy().reshape(60, 60)
        pi_grid = q_grid.argmax(dim=1).numpy().reshape(60, 60)

    c = ax1.contourf(G_TH, G_THD, v_grid, levels=30, cmap='viridis')
    plt.colorbar(c, ax=ax1, label='Max Q(s, a) [State Value V(s)]')
    
    # Subsample for policy arrows (0: Force -10, 1: 0, 2: Force +10)
    step = 4
    U_arrows = np.zeros_like(G_TH[::step, ::step])
    V_arrows = np.zeros_like(G_THD[::step, ::step])
    pi_sub = pi_grid[::step, ::step]
    U_arrows[pi_sub == 0] = -1.0 # Push Left
    U_arrows[pi_sub == 2] = +1.0 # Push Right
    ax1.quiver(G_TH[::step, ::step], G_THD[::step, ::step], U_arrows, V_arrows, color='white', alpha=0.8, scale=25)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.6, label='Upright Target (θ=0)')
    ax1.axvspan(-0.2, 0.2, color='red', alpha=0.15, label='Upright Goal Cone (|θ| < 0.2)')
    ax1.set_title("1. Value Landscape V(θ, θ̇) & Policy Vector Field", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Angle θ (rad) [-π: Down, 0: Upright]")
    ax1.set_ylabel("Angular Velocity θ̇ (rad/s)")
    ax1.legend(loc='upper right', fontsize=8.5)

    # --- PANEL 2: Replay Buffer Phase Space Coverage ---
    ax2 = axes[0, 1]
    buf_states = agent.replay.states[:len(agent.replay)]
    buf_rewards = agent.replay.rewards[:len(agent.replay)]
    # cos(th) is col 0, sin(th) is col 1 -> theta = arctan2(sin, cos)
    buf_thetas = np.arctan2(buf_states[:, 1], buf_states[:, 0])
    buf_th_dots = buf_states[:, 2]

    # Plot 2D histogram density of explored transitions
    h, xedges, yedges = np.histogram2d(buf_thetas, buf_th_dots, bins=[50, 50], range=[[-np.pi, np.pi], [-8, 8]])
    im = ax2.imshow(np.log1p(h.T), origin='lower', extent=[-np.pi, np.pi, -8, 8], aspect='auto', cmap='plasma')
    plt.colorbar(im, ax=ax2, label='Log Density (Exploration Coverage)')
    
    # Highlight reward transitions in bright cyan
    pos_mask = buf_rewards > 0.0
    if np.any(pos_mask):
        ax2.scatter(buf_thetas[pos_mask], buf_th_dots[pos_mask], color='cyan', s=8, alpha=0.7, label=f'Goal Rewards ({np.sum(pos_mask)} pts)')
        ax2.legend(loc='upper right', fontsize=8.5)

    ax2.axvline(0, color='white', linestyle='--', alpha=0.6)
    ax2.set_title("2. Replay Buffer State Coverage (Phase Space Density)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Angle θ (rad)")
    ax2.set_ylabel("Angular Velocity θ̇ (rad/s)")

    # --- PANEL 3: Warmstart Perturbation Magnitude Delta Q ---
    ax3 = axes[1, 0]
    eps_arr = np.arange(1, episodes + 1)
    ax3.plot(eps_arr, delta_q_history, color='crimson', linewidth=1.5, label='Mean |ΔQ| from Warmstart')
    ax3.axhspan(1.0, 5.0, color='green', alpha=0.15, label='Goldilocks Zone (1.0 - 5.0)')
    ax3.axhline(15.0, color='darkred', linestyle=':', label='Destructive Perturbation Threshold (15.0)')
    ax3.set_title("3. Warmstart Perturbation Shift (|ΔQ| across Eval Grid)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Mean |ΔQ| (Q_after - Q_before)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8.5)

    # --- PANEL 4: Effective Sample Size (ESS) & W_prior Dynamics ---
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    p1 = ax4.plot(eps_arr, ess_history, color='royalblue', linewidth=1.8, label='Effective Sample Size (ESS)')
    p2 = ax4_twin.plot(eps_arr, w_prior_history, color='darkorange', linewidth=1.8, linestyle='--', label='Prior Mass W_prior')
    ax4_twin.axhline(0.05, color='gray', linestyle=':', label='Prior Floor W_min = 0.05')
    
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=8.5)
    ax4.set_title("4. Posterior Weight Dispersion (ESS & W_prior)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Effective Sample Size (N_eff)", color='royalblue')
    ax4_twin.set_ylabel("Total Prior Mass (W_prior)", color='darkorange')
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"DP-DQN Diagnostic Suite: {sampler_type.upper()} on Cart-Pole Swing-Up (Ep 1-{episodes})", fontsize=14, y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_img_path)), exist_ok=True)
    plt.savefig(out_img_path, dpi=150, bbox_inches='tight')
    print(f"Saved complete diagnostic dashboard to: {out_img_path}")

if __name__ == "__main__":
    out_file = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86/dp_dqn_diagnostic_dashboard.png"
    run_diagnostic(episodes=150, seed=42, hidden_dim=64, use_layer_norm=False, sampler_type="dirichlet", alpha=3.0, out_img_path=out_file)
