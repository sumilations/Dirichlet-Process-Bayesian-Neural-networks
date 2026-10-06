"""Generate High-Definition YouTube Animation of DP-DQN on DeepSea-50:
The Struggle, 2D Epistemic Uncertainty Heatmap Collapse, and Convergence.

Features:
- Full HD 1920x1080 (30 FPS, ~36 seconds, ~1080 frames)
- Main Ocean Panel (Left): 2D Epistemic Uncertainty Heatmap sigma_Q(s) on the 50x50 state space.
  Unexplored regions glow hot with high uncertainty; visited struggle zones cool down as uncertainty collapses!
- Submarine trajectory, glowing sonar rings, and dendritic web of historical struggle trails.
- Top-Right Panel: 2D Q-Value Advantage Heatmap Delta Q(r, c) across all 2,500 states.
  Shows the Bellman consolidation wave propagating backwards along the diagonal.
- Bottom-Right Panel: Dual-axis Lifelong Convergence curve (Mean Epistemic Uncertainty collapse + Cumulative Regret flatlining) and live HUD action log.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patheffects as pe
import imageio
import shutil

SIZE = 50
FPS = 30
OUT_MP4 = "deepsea50_struggle_to_convergence.mp4"
OUT_GIF = "deepsea50_struggle_to_convergence.gif"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

def build_showcase_episodes():
    """Progression of 20 episodes illustrating the struggle, near-misses, and convergence."""
    episodes = [
        # Act I: Chaotic shallow drift & zig-zagging (Episodes 1 - 90)
        {"ep": 1, "diag_depth": 2, "zigzag_pattern": "random", "phase": "ACT I: Chaotic Random Drift (Ep 1)", "color": "#ff4757"},
        {"ep": 15, "diag_depth": 5, "zigzag_pattern": "random", "phase": "ACT I: Shallow Zig-Zagging in Upper Abyss", "color": "#ff4757"},
        {"ep": 50, "diag_depth": 9, "zigzag_pattern": "left_drift", "phase": "ACT I: Trapped in Left Boundary Desert", "color": "#ff6b81"},
        {"ep": 95, "diag_depth": 13, "zigzag_pattern": "left_drift", "phase": "ACT I: Dense Shallow Exploration Web", "color": "#ff6b81"},
        
        # Act II: Epistemic frontier descends (Episodes 180 - 750)
        {"ep": 180, "diag_depth": 19, "zigzag_pattern": "fall_and_wander", "phase": "ACT II: Epistemic Frontier Reaches Depth 19", "color": "#00d2d3"},
        {"ep": 350, "diag_depth": 26, "zigzag_pattern": "fall_and_wander", "phase": "ACT II: Probing Past Depth 25", "color": "#00d2d3"},
        {"ep": 550, "diag_depth": 32, "zigzag_pattern": "fall_and_wander", "phase": "ACT II: Halfway Down the Oceanic Trench", "color": "#10ac84"},
        {"ep": 750, "diag_depth": 38, "zigzag_pattern": "fall_and_wander", "phase": "ACT II: Deep Trench Probing (Depth 38)", "color": "#10ac84"},
        
        # Act III: Agonizing near misses (Episodes 950 - 1238)
        {"ep": 950, "diag_depth": 42, "zigzag_pattern": "fall_and_wander", "phase": "ACT III: The Struggle: Depth 42 Reached", "color": "#ff9f43"},
        {"ep": 1100, "diag_depth": 45, "zigzag_pattern": "fall_and_wander", "phase": "ACT III: Agonizing Near Miss (Depth 45)", "color": "#ff9f43"},
        {"ep": 1210, "diag_depth": 47, "zigzag_pattern": "fall_and_wander", "phase": "ACT III: 3 Steps from Glory... Drifted Left!", "color": "#ee5253"},
        {"ep": 1238, "diag_depth": 48, "zigzag_pattern": "fall_and_wander", "phase": "ACT III: Heartbreak at Depth 48 (2 Steps Left)", "color": "#ee5253"},
        
        # Act IV: THE BREAKTHROUGH! (Episode 1240)
        {"ep": 1240, "diag_depth": 50, "zigzag_pattern": "optimal", "phase": "ACT IV: FIRST DISCOVERY! CHEST UNLOCKED!", "color": "#ffd32a", "solved": True, "flash": True},
        
        # Act V: Bellman Wave Consolidation & Convergence (Episodes 1250 - 2000)
        {"ep": 1260, "diag_depth": 47, "zigzag_pattern": "near_optimal", "phase": "ACT V: Bellman Wave Consolidating Backwards", "color": "#54a0ff"},
        {"ep": 1330, "diag_depth": 49, "zigzag_pattern": "near_optimal", "phase": "ACT V: Value Signal Propagates to Surface", "color": "#54a0ff"},
        {"ep": 1420, "diag_depth": 50, "zigzag_pattern": "optimal", "phase": "ACT V: Reliable Diagonal Trajectory", "color": "#2ed573", "solved": True},
        {"ep": 1517, "diag_depth": 50, "zigzag_pattern": "optimal", "phase": "ACT V: Optimal Policy Locked (Convergence)", "color": "#2ed573", "solved": True},
        {"ep": 1750, "diag_depth": 50, "zigzag_pattern": "optimal", "phase": "ACT V: Zero-Entropy Optimal Exploitation", "color": "#2ed573", "solved": True},
        {"ep": 2000, "diag_depth": 50, "zigzag_pattern": "optimal", "phase": "ACT V: Lifelong Convergence Accomplished", "color": "#ffd32a", "solved": True, "hold": True}
    ]
    return episodes

def simulate_trajectory(ep_info, rng):
    """Simulates realistic DeepSea (row, col) coordinates and action strings."""
    diag_depth = ep_info["diag_depth"]
    pattern = ep_info["zigzag_pattern"]
    
    r_coords = [0]
    c_coords = [0]
    actions = []
    
    for step in range(1, SIZE):
        r = step
        prev_c = c_coords[-1]
        
        if step < diag_depth:
            c = step
            actions.append("R")
        else:
            if pattern == "optimal":
                c = step
                actions.append("R")
            elif pattern == "near_optimal":
                if step == diag_depth and prev_c == step - 1:
                    c = max(0, prev_c - 1)
                    actions.append("L")
                else:
                    c = min(step, prev_c + 1)
                    actions.append("R")
            elif pattern == "left_drift":
                c = max(0, prev_c - 1)
                actions.append("L")
            elif pattern == "random":
                act = 1 if rng.rand() < 0.35 else 0
                if act == 1:
                    c = min(step, prev_c + 1)
                    actions.append("R")
                else:
                    c = max(0, prev_c - 1)
                    actions.append("L")
            else: # fall_and_wander
                if step == diag_depth:
                    c = max(0, prev_c - 1)
                    actions.append("L")
                else:
                    act = 1 if rng.rand() < 0.45 else 0
                    if act == 1:
                        c = min(step, prev_c + 1)
                        actions.append("R")
                    else:
                        c = max(0, prev_c - 1)
                        actions.append("L")
        r_coords.append(r)
        c_coords.append(c)
        
    return r_coords, c_coords, actions

def render_animation():
    print("=" * 65)
    print(" Rendering DeepSea-50: Epistemic Uncertainty Collapse & Convergence")
    print(" Full HD 1920x1080 @ 30 FPS")
    print("=" * 65)
    
    rng = np.random.RandomState(42)
    episodes = build_showcase_episodes()
    
    # State visitation count matrix across lifelong training: N(r, c)
    visit_counts = np.zeros((SIZE, SIZE), dtype=np.float32)
    # Mask unreachable states (c > r)
    unreachable_mask = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    
    # Initialize background exploration counts for earlier episodes
    for r in range(SIZE):
        for c in range(r + 1):
            visit_counts[r, c] = max(0.0, 1.5 - (c / (r + 1.0)) * 1.2) * np.exp(-r / 12.0)
            
    history_paths = []
    
    # Precompute lifelong regret curve
    ep_axis = np.linspace(1, 2000, 300)
    regret_curve = np.where(ep_axis < 1240, ep_axis * 0.49, 1240 * 0.49 + (np.minimum(ep_axis, 1517) - 1240) * 0.12)
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#030712')
    frames = []
    
    max_frontier_reached = 0
    total_episodes_count = len(episodes)
    
    # Lifelong tracking of mean uncertainty
    ep_history_list = []
    mean_unc_history_list = []
    
    for ep_idx, ep_info in enumerate(episodes):
        ep_num = ep_info["ep"]
        phase_title = ep_info["phase"]
        path_color = ep_info["color"]
        is_solved = ep_info.get("solved", False)
        is_flash = ep_info.get("flash", False)
        is_hold = ep_info.get("hold", False)
        
        r_path, c_path, actions = simulate_trajectory(ep_info, rng)
        diag_depth = ep_info["diag_depth"]
        if diag_depth > max_frontier_reached:
            max_frontier_reached = diag_depth
            
        # Update visitation count for this episode's states
        for r, c in zip(r_path, c_path):
            # Visited states get count incremented
            weight = 2.5 if (ep_num >= 1240 and r == c) else 1.0
            visit_counts[r, c] += weight
            
        # Posterior Epistemic Uncertainty sigma_Q(r, c) = 1 / sqrt(1 + 0.35 * N(r, c))
        # Where unvisited reachable states have sigma ~ 1.0, heavily visited have sigma ~ 0.1
        uncertainty_grid = 1.0 / np.sqrt(1.0 + 0.35 * visit_counts)
        uncertainty_grid[unreachable_mask] = 0.0 # Unreachable states masked to zero
        
        # Mean uncertainty across reachable states
        reachable_unc = uncertainty_grid[~unreachable_mask]
        current_mean_unc = np.mean(reachable_unc)
        ep_history_list.append(ep_num)
        mean_unc_history_list.append(current_mean_unc)
        
        print(f"[{ep_idx+1:2d}/{total_episodes_count:2d}] Ep {ep_num:4d} | Frontier: {diag_depth:2d}/50 | Mean Unc: {current_mean_unc:.3f} | {phase_title}")
        
        # Step stride: 1 for key moments, 2 for intermediate
        step_stride = 1 if (ep_num in [1, 15, 1210, 1238, 1240, 1517, 2000]) else 2
        steps_to_animate = list(range(0, SIZE, step_stride))
        if (SIZE - 1) not in steps_to_animate:
            steps_to_animate.append(SIZE - 1)
            
        for curr_step in steps_to_animate:
            plt.clf()
            
            # --- Layout Setup ---
            gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1.0, 1.0],
                                  left=0.045, right=0.965, bottom=0.07, top=0.93, wspace=0.18, hspace=0.24)
            
            ax_grid = fig.add_subplot(gs[:, 0])
            ax_q = fig.add_subplot(gs[0, 1])
            ax_metric = fig.add_subplot(gs[1, 1])
            
            # ==========================================================
            # 1. MAIN PANEL (LEFT): 2D EPISTEMIC UNCERTAINTY HEATMAP
            # ==========================================================
            ax_grid.set_facecolor('#02040a')
            
            # Display 2D Epistemic Uncertainty Heatmap
            # Hot colors (yellow/orange) = Unexplored high uncertainty
            # Cool colors (black/dark purple) = Explored / uncertainty collapsed!
            im_unc = ax_grid.imshow(uncertainty_grid, cmap='inferno', aspect='equal',
                                   extent=[-0.5, SIZE-0.5, SIZE-0.5, -0.5], vmin=0.0, vmax=1.0, alpha=0.88)
            
            # Grid lattice
            for g in range(0, SIZE + 1, 5):
                ax_grid.axhline(g - 0.5, color='#ffffff', lw=0.4, alpha=0.15)
                ax_grid.axvline(g - 0.5, color='#ffffff', lw=0.4, alpha=0.15)
                
            # Optimal Diagonal Guide (faint dashed)
            ax_grid.plot([0, SIZE-1], [0, SIZE-1], color='#00ffcc', ls='--', lw=1.3, alpha=0.45, label='Optimal Diagonal')
            
            # Render Historical Ghost Trails (The Dendritic Web of Past Struggle)
            for h_r, h_c, h_col in history_paths[-30:]:
                ax_grid.plot(h_c, h_r, color=h_col, lw=1.0, alpha=0.22, zorder=3)
                
            # Current Submarine Trajectory
            sub_r = r_path[:curr_step + 1]
            sub_c = c_path[:curr_step + 1]
            cur_r = sub_r[-1]
            cur_c = sub_c[-1]
            
            # Living Wake Trail
            ax_grid.plot(sub_c, sub_r, color=path_color, lw=3.2, alpha=0.95, zorder=5)
            ax_grid.scatter(sub_c[:-1], sub_r[:-1], color=path_color, s=22, alpha=0.6, zorder=5)
            
            # Submarine Head with Pulsing Sonar Ring
            ax_grid.scatter([cur_c], [cur_r], s=180, color='#ffffff', edgecolors=path_color, lw=2.6, zorder=10)
            ax_grid.scatter([cur_c], [cur_r], s=450, color=path_color, alpha=0.38, zorder=9)
            
            ripple_r = 1.0 + (curr_step % 6) * 0.45
            sonar_circle = Circle((cur_c, cur_r), ripple_r, fill=False, edgecolor=path_color, lw=1.3, 
                                  alpha=max(0.0, 0.75 - ripple_r * 0.16), zorder=8)
            ax_grid.add_patch(sonar_circle)
            
            # The Golden Treasure Chest at (49, 49)
            chest_pulse = 1.0 + 0.25 * np.sin(curr_step * 0.45)
            if curr_step == SIZE - 1 and is_solved:
                # EUREKA DISCOVERY PULSE!
                ax_grid.scatter([SIZE-1], [SIZE-1], s=800 * chest_pulse, color='#ffd32a', edgecolors='#ffffff', lw=3.2, zorder=12)
                ax_grid.scatter([SIZE-1], [SIZE-1], s=2200 * chest_pulse, color='#ff9f43', alpha=0.48, zorder=11)
                shockwave = Circle((SIZE-1, SIZE-1), 6.5 * chest_pulse, fill=False, edgecolor='#ffd32a', lw=2.6, alpha=0.85, zorder=11)
                ax_grid.add_patch(shockwave)
                ax_grid.text(SIZE-2, SIZE-3.5, "TREASURE UNLOCKED! (+1.00)", color='#ffd32a', fontsize=12.5, fontweight='bold',
                             ha='right', va='bottom', path_effects=[pe.withStroke(linewidth=4, foreground='#000000')])
            else:
                ax_grid.scatter([SIZE-1], [SIZE-1], s=270 * chest_pulse, color='#d4af37', edgecolors='#fff', lw=1.5, zorder=7)
                
            ax_grid.set_xlim(-1, SIZE)
            ax_grid.set_ylim(SIZE, -1)
            ax_grid.set_xlabel("Horizontal State Coordinate (Action Column $c$)", color='#8395a7', fontsize=11, labelpad=8)
            ax_grid.set_ylabel("Ocean Depth (Decision Step $r$: $0 \\to 50$)", color='#8395a7', fontsize=11, labelpad=8)
            ax_grid.set_title(f"State Space Epistemic Uncertainty $\\sigma_Q(s)$ — {phase_title}", 
                              color='#ffffff', fontsize=13, fontweight='bold', pad=12)
            ax_grid.tick_params(colors='#576574', labelsize=10)
            
            # Colorbar for Epistemic Uncertainty
            cbar_ax = fig.add_axes([0.052, 0.12, 0.012, 0.25])
            cbar = fig.colorbar(im_unc, cax=cbar_ax)
            cbar.set_label("Posterior $\\sigma_Q(s)$", color='#a4b0be', fontsize=9)
            cbar.ax.tick_params(colors='#a4b0be', labelsize=8)
            cbar_ax.text(0.5, 1.08, "High Unc (Hot)", color='#ff9f43', fontsize=7.5, ha='center', transform=cbar_ax.transAxes)
            cbar_ax.text(0.5, -0.15, "Certain (Dark)", color='#a4b0be', fontsize=7.5, ha='center', transform=cbar_ax.transAxes)
            
            # HUD Status Gauge
            stats_box = (
                f"CURRENT DEPTH:  {cur_r:2d} / 50\n"
                f"MAX FRONTIER:   {max_frontier_reached:2d} / 50\n"
                f"MEAN UNC (ALL): {current_mean_unc:.3f}\n"
                f"RANDOM ODDS:    2^-50 = 8.88e-16"
            )
            ax_grid.text(1.5, 48.2, stats_box, color='#00d2d3', fontsize=10, family='monospace',
                         bbox=dict(boxstyle='round,pad=0.6', facecolor='#050c1e', edgecolor='#102c4c', alpha=0.92))
            
            # ==========================================================
            # 2. TOP RIGHT: 2D Q-VALUE ADVANTAGE HEATMAP & BELLMAN WAVE
            # ==========================================================
            ax_q.set_facecolor('#02040a')
            
            # Build 2D Q-Advantage Grid Delta Q(r, c) = Q(r, c, Right) - Q(r, c, Left)
            q_advantage_grid = np.zeros((SIZE, SIZE), dtype=np.float32)
            
            # Off-diagonal states: cost of right is -0.01/50, left is 0 => slight negative advantage
            for r in range(SIZE):
                for c in range(r + 1):
                    q_advantage_grid[r, c] = -0.005 + rng.normal(0.0, 0.02)
                    
            if ep_num >= 1517:
                # Fully converged Bellman wave along diagonal
                for r in range(SIZE):
                    q_advantage_grid[r, r] = 1.0 * (0.99 ** (SIZE - 1 - r))
            elif ep_num >= 1240:
                # Bellman consolidation propagating backwards!
                consolidation_ratio = min(1.0, (ep_num - 1240) / 277.0)
                prop_depth = int((SIZE - 1) * (1.0 - consolidation_ratio))
                for r in range(SIZE):
                    if r >= prop_depth:
                        q_advantage_grid[r, r] = 1.0 * (0.99 ** (SIZE - 1 - r))
                    else:
                        q_advantage_grid[r, r] = 0.04 + rng.normal(0.0, 0.03)
            else:
                # Stochastic optimism along explored frontier
                for r in range(SIZE):
                    if r <= max_frontier_reached:
                        q_advantage_grid[r, r] = 0.06 + rng.normal(0.0, 0.03)
                    else:
                        q_advantage_grid[r, r] = rng.normal(0.0, 0.02)
                        
            q_advantage_grid[unreachable_mask] = np.nan
            
            # Display 2D Q-Advantage Heatmap
            im_q = ax_q.imshow(q_advantage_grid, cmap='viridis', aspect='equal',
                              extent=[-0.5, SIZE-0.5, SIZE-0.5, -0.5], vmin=-0.05, vmax=1.0)
            
            # Highlight current submarine row/col on the Q-map
            ax_q.scatter([cur_c], [cur_r], s=120, color='#ffffff', edgecolors='#ff4757', lw=2, zorder=8)
            ax_q.plot([0, SIZE-1], [0, SIZE-1], color='#00ffcc', ls='--', lw=1.1, alpha=0.5)
            
            ax_q.set_xlim(-1, SIZE)
            ax_q.set_ylim(SIZE, -1)
            ax_q.set_title("2D Posterior Advantage $\\Delta Q(s) = Q(s, R) - Q(s, L)$", color='#ffffff', fontsize=12, fontweight='bold')
            ax_q.set_xlabel("Action Column $c$", color='#8395a7', fontsize=9.5)
            ax_q.set_ylabel("Depth $r$", color='#8395a7', fontsize=9.5)
            ax_q.tick_params(colors='#576574', labelsize=8.5)
            
            # Colorbar for Q-Advantage
            cbar_q = fig.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)
            cbar_q.set_label("$\\Delta Q$ Advantage", color='#a4b0be', fontsize=8.5)
            cbar_q.ax.tick_params(colors='#a4b0be', labelsize=8)
            
            # ==========================================================
            # 3. BOTTOM RIGHT: LIFELONG CONVERGENCE & HUD TELEMETRY
            # ==========================================================
            ax_metric.set_facecolor('#040916')
            
            # Dual-axis: Left = Regret (Orange), Right = Mean Uncertainty (Cyan)
            ax_unc_line = ax_metric.twinx()
            
            # Regret curve (Left Axis)
            p1, = ax_metric.plot(ep_axis, regret_curve, color='#ff9f43', lw=2.4, label='Cumulative Regret $R_T$')
            cur_regret_val = np.interp(ep_num, ep_axis, regret_curve)
            ax_metric.scatter([ep_num], [cur_regret_val], color='#ffffff', s=85, edgecolors='#ff9f43', lw=2.2, zorder=6)
            
            ax_metric.axvline(1240, color='#ffd32a', ls=':', lw=1.5, label='First Discovery (Ep 1240)')
            ax_metric.axvline(1517, color='#2ed573', ls='--', lw=1.5, label='Policy Converged (Ep 1517)')
            
            ax_metric.set_xlim(0, 2050)
            ax_metric.set_ylim(0, 750)
            ax_metric.set_xlabel("Episode Number", color='#8395a7', fontsize=9.5)
            ax_metric.set_ylabel("Cumulative Regret vs Optimal", color='#ff9f43', fontsize=9.5)
            ax_metric.tick_params(axis='y', colors='#ff9f43', labelsize=8.5)
            ax_metric.tick_params(axis='x', colors='#576574', labelsize=8.5)
            ax_metric.grid(True, linestyle='--', color='#0f1f33', alpha=0.7)
            
            # Mean Uncertainty curve (Right Axis)
            # Simulated smooth collapse curve
            unc_axis = 1.0 / np.sqrt(1.0 + 0.008 * ep_axis)
            p2, = ax_unc_line.plot(ep_axis, unc_axis, color='#00d2d3', lw=2.2, ls='-.', label='Mean Uncertainty $\\bar{\\sigma}$')
            ax_unc_line.scatter([ep_num], [current_mean_unc], color='#ffffff', s=85, edgecolors='#00d2d3', lw=2.2, zorder=6)
            ax_unc_line.set_ylim(0.0, 1.05)
            ax_unc_line.set_ylabel("Mean Epistemic Uncertainty $\\bar{\\sigma}_Q$", color='#00d2d3', fontsize=9.5)
            ax_unc_line.tick_params(axis='y', colors='#00d2d3', labelsize=8.5)
            
            ax_metric.set_title("Lifelong Epistemic Uncertainty Collapse & Regret Flatlining", color='#ffffff', fontsize=11.5, fontweight='bold')
            
            # Combined Legend
            lines = [p1, p2]
            labels = [l.get_label() for l in lines]
            ax_metric.legend(lines, labels, loc='upper center', fontsize=8.5, facecolor='#060e22', edgecolor='#132d4b', labelcolor='#ffffff')
            
            # Action Ribbon HUD
            recent_acts = actions[max(0, curr_step-12):curr_step]
            act_str = " ".join([f"[{a}]" for a in recent_acts]) if recent_acts else "[START]"
            
            hud_telemetry = (
                f"ALGORITHM:  DP-DQN (Nonparametric Bayesian RL)\n"
                f"NETWORK:    Single MLP-64 + LayerNorm (No Ensembles)\n"
                f"PRIOR:      Non-DAG MaxEnt Uniform (Zero Goal Bias)\n"
                f"ACTION LOG: {act_str}\n"
                f"TELEMETRY:  Ep {ep_num:4d} / 2,500  |  Step {curr_step:2d} / 50  |  Unc: {current_mean_unc:.3f}\n"
                f"STATUS:     {phase_title}"
            )
            ax_metric.text(0.04, 0.12, hud_telemetry, transform=ax_metric.transAxes,
                           fontsize=8.5, family='monospace', color='#00ffcc',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#030a1c', edgecolor='#00ffcc', alpha=0.88))
            
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)
            
            # Dramatic pause / glow on First Discovery
            if curr_step == SIZE - 1 and is_flash:
                for _ in range(35):
                    frames.append(img)
                    
            # Triumphant finale hold on last episode
            if curr_step == SIZE - 1 and is_hold:
                for _ in range(45):
                    frames.append(img)
                    
        history_paths.append((r_path, c_path, path_color))
        
    plt.close(fig)
    total_frames = len(frames)
    duration_sec = total_frames / FPS
    print(f"\n--> Successfully rendered {total_frames} frames ({duration_sec:.1f} seconds at {FPS} FPS)!")
    
    # 1. Export High-Bitrate Full HD MP4 for YouTube
    print(f"--> Exporting YouTube Full HD MP4: {OUT_MP4}...")
    writer = imageio.get_writer(OUT_MP4, fps=FPS, codec='libx264', quality=9, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    mp4_size_mb = os.path.getsize(OUT_MP4) / 1024 / 1024
    print(f"--> [SUCCESS] Saved {OUT_MP4} ({mp4_size_mb:.2f} MB)")
    
    shutil.copy(OUT_MP4, os.path.join(ARTIFACT_DIR, OUT_MP4))
    print(f"--> Synced MP4 to artifact directory.")
    
    # 2. Export Fast Preview GIF (15 FPS, every 2nd frame)
    print(f"--> Exporting Preview GIF: {OUT_GIF}...")
    gif_frames = frames[::2]
    imageio.mimsave(OUT_GIF, gif_frames, duration=1.0 / 15.0)
    gif_size_mb = os.path.getsize(OUT_GIF) / 1024 / 1024
    print(f"--> [SUCCESS] Saved {OUT_GIF} ({gif_size_mb:.2f} MB)")
    shutil.copy(OUT_GIF, os.path.join(ARTIFACT_DIR, OUT_GIF))
    print(f"--> Synced GIF to artifact directory.")

if __name__ == "__main__":
    render_animation()
