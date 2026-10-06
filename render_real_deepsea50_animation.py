"""Render High-Definition Presentation Video from 100% Real Logged DP-DQN Training Data on DeepSea-50.

Reads real_deepsea50_q_log.npz:
- True neural network output Delta Q_theta(r, c) on every cell
- True trajectories taken by the agent
- True episodic hitting time, discovery, and convergence
- Exports full 1080p MP4 for YouTube and animated GIF preview
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patheffects as pe
import imageio
import shutil

SIZE = 50
FPS = 30
LOG_FILE = "real_deepsea50_q_log.npz"
OUT_MP4 = "real_deepsea50_dpdqn_youtube.mp4"
OUT_GIF = "real_deepsea50_dpdqn_youtube.gif"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

def render_from_real_data():
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found! Please wait for training to complete.")
        return
        
    print("=" * 70)
    print(f" Loading Real DeepSea-50 Data from {LOG_FILE}...")
    data = np.load(LOG_FILE, allow_pickle=True)
    first_discovery_ep = int(data["first_discovery_ep"])
    solved_ep = int(data["solved_ep"])
    total_time = float(data["total_time"])
    snapshots = data["snapshots"]
    total_snaps = len(snapshots)
    print(f" Loaded {total_snaps} real snapshots! Discovery: Ep {first_discovery_ep} | Solved: Ep {solved_ep}")
    print("=" * 70)
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#02040c')
    frames = []
    
    # Track historical ghost trails from real runs
    history_trails = []
    unreachable = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    
    for snap_idx, snap in enumerate(snapshots):
        ep_num = int(snap["episode"])
        trajectory = snap["trajectory"] # (steps, 2)
        actions = snap["actions"]
        delta_q = snap["delta_q"] # (50, 50)
        v_val = snap["v_val"]
        ep_ret = float(snap["ep_return"])
        cum_regret = float(snap["cum_regret"])
        is_solved = bool(snap["is_solved"])
        
        is_discovery = (ep_num == first_discovery_ep)
        is_converged = (ep_num >= solved_ep and is_solved)
        is_finale = (snap_idx == total_snaps - 1)
        
        # Color coding by era
        if ep_num < 200:
            phase_text = f"ACT I: CHAOTIC EXPLORATION & STRUGGLE (Ep {ep_num})"
            glow_theme = "#ff4757"
            trail_col = "#ff4757"
            timeline_pos = 0.12
        elif ep_num < 1000:
            phase_text = f"ACT II: EPISTEMIC DRILLING DOWN THE TRENCH (Ep {ep_num})"
            glow_theme = "#00d2d3"
            trail_col = "#00d2d3"
            timeline_pos = 0.40
        elif ep_num < first_discovery_ep:
            phase_text = f"ACT III: AGONIZING NEAR-MISSES IN THE ABYSS (Ep {ep_num})"
            glow_theme = "#ff9f43"
            trail_col = "#ff9f43"
            timeline_pos = 0.68
        elif is_discovery:
            phase_text = f"ACT IV: THE HISTORIC BREAKTHROUGH! (Ep {ep_num}) CHEST UNLOCKED!"
            glow_theme = "#ffd32a"
            trail_col = "#ffd32a"
            timeline_pos = 0.82
        elif ep_num < solved_ep:
            phase_text = f"ACT V: BELLMAN CONSOLIDATION WAVE PROPAGATING BACK (Ep {ep_num})"
            glow_theme = "#54a0ff"
            trail_col = "#54a0ff"
            timeline_pos = 0.90
        else:
            phase_text = f"ACT V: CONVERGED OPTIMAL POLICY & MASTERY (Ep {ep_num})"
            glow_theme = "#2ed573"
            trail_col = "#2ed573"
            timeline_pos = 0.98
            
        print(f"[{snap_idx+1:2d}/{total_snaps:2d}] Ep {ep_num:4d} | Return: {ep_ret:.3f} | {phase_text}")
        
        # Step stride: slow down for discovery and finale
        step_stride = 1 if (is_discovery or is_finale or ep_num in [1, 25]) else 2
        steps_to_animate = list(range(0, len(trajectory), step_stride))
        if (len(trajectory) - 1) not in steps_to_animate:
            steps_to_animate.append(len(trajectory) - 1)
            
        # Clean Delta Q for visualization (mask unreachable)
        masked_delta_q = delta_q.copy()
        masked_delta_q[unreachable] = np.nan
        
        for curr_step in steps_to_animate:
            plt.clf()
            
            # --- FULL-SCREEN CINEMATIC CANVAS ---
            gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.65], left=0.03, right=0.97, bottom=0.06, top=0.92, wspace=0.10)
            ax_main = fig.add_subplot(gs[0, 0])
            ax_info = fig.add_subplot(gs[0, 1])
            
            # ==============================================================
            # 1. MAIN GRID: REAL NEURAL NETWORK Q-VALUE HEATMAP
            # ==============================================================
            ax_main.set_facecolor('#01030a')
            
            # Colormap: Viridis or Coolwarm for real Q-advantage
            # Real Q-values: negative is dark/purple, positive advantage along diagonal lights up!
            im_q = ax_main.imshow(masked_delta_q, cmap='viridis', aspect='equal',
                                 extent=[-0.5, SIZE-0.5, SIZE-0.5, -0.5], vmin=-0.05, vmax=1.0)
            
            # Faint grid lattice
            for g in range(0, SIZE + 1, 5):
                ax_main.axhline(g - 0.5, color='#ffffff', lw=0.35, alpha=0.12)
                ax_main.axvline(g - 0.5, color='#ffffff', lw=0.35, alpha=0.12)
                
            # Optimal Diagonal Guide (faint cyan dashed)
            ax_main.plot([0, SIZE-1], [0, SIZE-1], color='#00ffcc', ls='--', lw=1.2, alpha=0.35)
            
            # Render Real Historical Ghost Trails (Past Struggles)
            for h_coords, h_col, h_a in history_trails[-40:]:
                h_r = h_coords[:, 0]
                h_c = h_coords[:, 1]
                ax_main.plot(h_c, h_r, color=h_col, lw=0.9, alpha=h_a, zorder=3)
                
            # Current Submarine Real Living Dive
            sub_coords = trajectory[:curr_step + 1]
            sub_r = sub_coords[:, 0]
            sub_c = sub_coords[:, 1]
            cur_r = sub_r[-1]
            cur_c = sub_c[-1]
            
            # Living Wake Trail
            ax_main.plot(sub_c, sub_r, color=trail_col, lw=3.6, alpha=0.95, zorder=6)
            ax_main.scatter(sub_c[:-1], sub_r[:-1], color=trail_col, s=24, alpha=0.55, zorder=6)
            
            # Forward Sonar Lighting Beam
            if curr_step < len(trajectory) - 1:
                sonar_pts = np.array([
                    [cur_c, cur_r],
                    [cur_c - 1.8, min(SIZE - 1, cur_r + 4.5)],
                    [cur_c + 1.8, min(SIZE - 1, cur_r + 4.5)]
                ])
                sonar_beam = Polygon(sonar_pts, closed=True, facecolor=trail_col, alpha=0.18, zorder=5)
                ax_main.add_patch(sonar_beam)
                
            # Submarine Head
            ax_main.scatter([cur_c], [cur_r], s=220, color='#ffffff', edgecolors=trail_col, lw=2.8, zorder=12)
            ax_main.scatter([cur_c], [cur_r], s=600, color=trail_col, alpha=0.40, zorder=11)
            
            # Pulsing Sonar Ring
            ring_r = 1.0 + (curr_step % 5) * 0.5
            ring = Circle((cur_c, cur_r), ring_r, fill=False, edgecolor=trail_col, lw=1.4,
                          alpha=max(0.0, 0.7 - ring_r * 0.15), zorder=10)
            ax_main.add_patch(ring)
            
            # The Golden Treasure Chest at (49, 49)
            pulse = 1.0 + 0.28 * np.sin(curr_step * 0.5)
            if curr_step >= SIZE - 1 and is_discovery:
                # SUPERNOVA EXPLOSION ON REAL DISCOVERY
                ax_main.scatter([SIZE-1], [SIZE-1], s=1200 * pulse, color='#ffd32a', edgecolors='#ffffff', lw=3.5, zorder=15)
                ax_main.scatter([SIZE-1], [SIZE-1], s=3500 * pulse, color='#ff9f43', alpha=0.55, zorder=14)
                sw1 = Circle((SIZE-1, SIZE-1), 7.0 * pulse, fill=False, edgecolor='#ffd32a', lw=3.0, alpha=0.85, zorder=14)
                sw2 = Circle((SIZE-1, SIZE-1), 14.0 * pulse, fill=False, edgecolor='#ff9f43', lw=1.8, alpha=0.60, zorder=14)
                ax_main.add_patch(sw1)
                ax_main.add_patch(sw2)
                ax_main.text(SIZE-2, SIZE-4.5, "★ FIRST DISCOVERY: CHEST UNLOCKED! ★", color='#ffd32a',
                             fontsize=13, fontweight='bold', ha='right', va='bottom',
                             path_effects=[pe.withStroke(linewidth=4.5, foreground='#000000')])
            elif is_converged and curr_step >= SIZE - 1:
                ax_main.scatter([SIZE-1], [SIZE-1], s=800 * pulse, color='#2ed573', edgecolors='#ffffff', lw=2.8, zorder=14)
                ax_main.scatter([SIZE-1], [SIZE-1], s=2000 * pulse, color='#00d2d3', alpha=0.45, zorder=13)
            else:
                ax_main.scatter([SIZE-1], [SIZE-1], s=300 * pulse, color='#d4af37', edgecolors='#ffffff', lw=1.5, zorder=8)
                
            ax_main.set_xlim(-1.5, SIZE + 0.5)
            ax_main.set_ylim(SIZE + 0.5, -1.5)
            ax_main.axis('off')
            
            # Subtle coordinate tags
            ax_main.text(0, -0.6, "SURFACE (0, 0)", color='#00d2d3', fontsize=9.5, fontweight='bold', family='monospace')
            ax_main.text(SIZE-1, SIZE+0.2, "ABYSS CHEST (49, 49)", color='#ffd32a', fontsize=9.5, fontweight='bold', family='monospace', ha='right')
            
            # ==============================================================
            # 2. RIGHT PANEL: LIVE TELEMETRY CARDS
            # ==============================================================
            ax_info.set_facecolor('#02040c')
            ax_info.axis('off')
            
            # Card 1: Live Episode & Real Coordinates
            card1 = (
                f"REAL DP-DQN TELEMETRY:\n"
                f"• EPISODE:         {ep_num:4d} / 1,537\n"
                f"• CURRENT DEPTH:   {cur_r:2d} / 50\n"
                f"• CURRENT COLUMN:  {cur_c:2d} / 50\n"
                f"• STEP IN DIVE:    {curr_step:2d} / 50\n"
                f"• RANDOM ODDS:     2^-50 ≈ 8.88 × 10^-16"
            )
            ax_info.text(0.05, 0.95, card1, transform=ax_info.transAxes,
                         color='#00ffcc', fontsize=11, family='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#060d22', edgecolor='#00ffcc', lw=1.2, alpha=0.92))
            
            # Card 2: Exact Neural Network Q-Values at Current State
            real_delta_at_cur = float(delta_q[cur_r, cur_c])
            real_v_at_cur = float(v_val[cur_r, cur_c])
            card2 = (
                f"NEURAL NET AT s=({cur_r},{cur_c}):\n"
                f"• Advantage ΔQ(s): {real_delta_at_cur:+.4f}\n"
                f"• Value V(s):       {real_v_at_cur:+.4f}\n"
                f"• Network State:    2-layer MLP-64\n"
                f"• Architecture:     LayerNorm + No Ensembles"
            )
            ax_info.text(0.05, 0.63, card2, transform=ax_info.transAxes,
                         color=glow_theme, fontsize=10.5, family='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#060d22', edgecolor=glow_theme, lw=1.2, alpha=0.92))
            
            # Card 3: Lifelong Regret & Real Performance
            card3 = (
                f"LIFELONG PERFORMANCE:\n"
                f"• Cumulative Regret: {cum_regret:.1f}\n"
                f"• Episode Return:    {ep_ret:.3f}\n"
                f"• Discovery Episode: {first_discovery_ep}\n"
                f"• Solved Episode:    {solved_ep}"
            )
            ax_info.text(0.05, 0.33, card3, transform=ax_info.transAxes,
                         color='#ff9f43', fontsize=10.5, family='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#060d22', edgecolor='#ff9f43', lw=1.2, alpha=0.92))
            
            # ==============================================================
            # TOP HEADER BAR
            # ==============================================================
            fig.text(0.03, 0.955, "DEEPSEA-50 (100% REAL NEURAL NET Q-VALUES)", color='#ffffff', fontsize=14.5, fontweight='bold')
            fig.text(0.44, 0.955, phase_text, color=glow_theme, fontsize=12.0, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='#000000')])
            fig.text(0.97, 0.955, "GOOGLE DEEPMIND RL", color='#747d8c', fontsize=11, family='monospace', ha='right')
            
            # ==============================================================
            # BOTTOM TIMELINE
            # ==============================================================
            timeline_y = 0.035
            fig.add_artist(plt.Line2D([0.05, 0.95], [timeline_y, timeline_y], color='#1e272e', lw=3.5))
            fig.add_artist(plt.Line2D([0.05, 0.05 + 0.90 * timeline_pos], [timeline_y, timeline_y], color=glow_theme, lw=4.0))
            fig.add_artist(Circle((0.05 + 0.90 * timeline_pos, timeline_y), 0.008, color='#ffffff', zorder=20, transform=fig.transFigure))
            
            fig.text(0.05, 0.015, "EP 1: CHAOTIC SEARCH", color='#747d8c', fontsize=8.5, family='monospace')
            fig.text(0.38, 0.015, "EP 600: DRILLING", color='#747d8c', fontsize=8.5, family='monospace')
            fig.text(0.68, 0.015, f"EP {first_discovery_ep}: DISCOVERY", color='#ffd32a', fontsize=8.5, family='monospace', fontweight='bold')
            fig.text(0.95, 0.015, f"EP {solved_ep}: CONVERGED", color='#2ed573', fontsize=8.5, family='monospace', ha='right', fontweight='bold')
            
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)
            
            if is_discovery and curr_step >= SIZE - 1:
                for _ in range(35): # 1.2s freeze-frame on real discovery
                    frames.append(img)
                    
            if is_finale and curr_step >= SIZE - 1:
                for _ in range(45): # 1.5s freeze-frame on final mastery
                    frames.append(img)
                    
        trail_a = 0.15 if ep_num < 1000 else 0.28
        history_trails.append((trajectory, trail_col, trail_a))
        
    plt.close(fig)
    total_frames = len(frames)
    dur = total_frames / FPS
    print(f"\n--> Successfully rendered {total_frames} frames ({dur:.1f}s @ {FPS} FPS) from REAL data!")
    
    # Export YouTube MP4
    print(f"--> Exporting YouTube Full HD MP4: {OUT_MP4}...")
    writer = imageio.get_writer(OUT_MP4, fps=FPS, codec='libx264', quality=9, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    mp4_mb = os.path.getsize(OUT_MP4) / 1024 / 1024
    print(f"--> [SUCCESS] Saved {OUT_MP4} ({mp4_mb:.2f} MB)")
    shutil.copy(OUT_MP4, os.path.join(ARTIFACT_DIR, OUT_MP4))
    
    # Export Preview GIF
    print(f"--> Exporting Preview GIF: {OUT_GIF}...")
    imageio.mimsave(OUT_GIF, frames[::2], duration=1.0 / 15.0)
    gif_mb = os.path.getsize(OUT_GIF) / 1024 / 1024
    print(f"--> [SUCCESS] Saved {OUT_GIF} ({gif_mb:.2f} MB)")
    shutil.copy(OUT_GIF, os.path.join(ARTIFACT_DIR, OUT_GIF))
    print(f"--> Finished! All files ready for YouTube.")

if __name__ == "__main__":
    render_from_real_data()
