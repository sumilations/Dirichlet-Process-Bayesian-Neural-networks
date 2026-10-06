"""Cinematic High-End Animation of DP-DQN Solving DeepSea-50.

Designed for YouTube presentations:
- 1080p Full HD (1920x1080 @ 30 FPS)
- Single unified cinematic canvas (no cramped subplots or academic tick marks)
- Glowing oceanic grid where each cell's ambient light reflects live Q-values / epistemic uncertainty
- Submarine with animated forward sonar beam and particle wake
- 60+ rapid-fire dive attempts creating an organic glowing dendritic web of struggle
- Dramatic slow-motion on the near-miss at depth 48 and the historic discovery at episode 1240
- Golden supernova shockwave upon unlocking the treasure chest
- Sleek modern sci-fi HUD with status timeline and odometer
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
OUT_MP4 = "deepsea50_cinematic_youtube.mp4"
OUT_GIF = "deepsea50_cinematic_youtube.gif"
ARTIFACT_DIR = "/Users/sumitvashishtha/.gemini/antigravity/brain/f3853f6f-e164-4df3-ac40-df17bbc1fc86"

def generate_cinematic_video():
    print("=" * 65)
    print(" Generating Cinematic DeepSea-50 Presentation Video (1080p @ 30 FPS)")
    print("=" * 65)
    
    rng = np.random.RandomState(42)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#02050e')
    frames = []
    
    # 50x50 Q-value grid and visitation grid
    q_grid = np.zeros((SIZE, SIZE), dtype=np.float32)
    visit_grid = np.zeros((SIZE, SIZE), dtype=np.float32)
    unreachable = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    
    # Historical struggle trails: store list of (path_r, path_c, color, alpha)
    history_trails = []
    
    # Define the 4 cinematic acts and episodes to render:
    # We will render ~40 struggle episodes rapidly, then slow down for the near-miss and discovery
    struggle_depths = [
        # Act 1: Chaotic surface zig-zags (episodes 1 to 100)
        (1, 2, "random", "#ff3838"), (5, 3, "random", "#ff3838"), (12, 4, "random", "#ff3838"),
        (25, 6, "drift", "#ff4d4d"), (40, 8, "drift", "#ff4d4d"), (60, 9, "drift", "#ff4d4d"),
        (85, 11, "drift", "#ff6b6b"), (110, 13, "drift", "#ff6b6b"),
        
        # Act 2: Epistemic exploration pushing deeper (episodes 150 to 900)
        (160, 16, "wander", "#00d2d3"), (220, 19, "wander", "#00d2d3"), (290, 22, "wander", "#00d2d3"),
        (370, 25, "wander", "#10ac84"), (460, 28, "wander", "#10ac84"), (560, 31, "wander", "#10ac84"),
        (670, 34, "wander", "#01a3a4"), (780, 37, "wander", "#01a3a4"), (890, 40, "wander", "#01a3a4"),
        
        # Act 3: Agonizing near misses (episodes 1000 to 1239)
        (1020, 43, "wander", "#ff9f43"), (1140, 45, "wander", "#ff9f43"),
        (1210, 47, "wander", "#ee5253"), (1238, 48, "near_miss", "#ff4757"), # Agonizing near miss!
        
        # Act 4: The Historic Discovery! (Episode 1240)
        (1240, 50, "optimal", "#ffd32a"),
        
        # Act 5: Convergence and Mastery (Episodes 1260 to 2000)
        (1270, 48, "near_optimal", "#54a0ff"), (1340, 50, "optimal", "#2ed573"),
        (1517, 50, "optimal", "#2ed573"), (2000, 50, "optimal", "#ffd32a")
    ]
    
    total_acts = len(struggle_depths)
    
    # Pre-render some historical faint background exploration
    for r in range(SIZE):
        for c in range(r + 1):
            visit_grid[r, c] = max(0.0, 1.0 - (c / (r + 1.0)) * 1.1) * np.exp(-r / 10.0)
            
    for act_idx, (ep_num, max_d, pattern, trail_col) in enumerate(struggle_depths):
        # Determine trajectory coordinates
        r_coords = [0]
        c_coords = [0]
        for step in range(1, SIZE):
            r = step
            prev_c = c_coords[-1]
            if step < max_d:
                c = step # on diagonal
            else:
                if pattern == "optimal":
                    c = step
                elif pattern == "near_miss":
                    # At depth 48, turns left!
                    if step == 48:
                        c = prev_c - 1
                    else:
                        c = max(0, prev_c - 1)
                elif pattern == "near_optimal":
                    if step >= 48:
                        c = prev_c - 1
                    else:
                        c = step
                elif pattern == "drift":
                    c = max(0, prev_c - 1)
                elif pattern == "random":
                    c = min(step, prev_c + 1) if rng.rand() < 0.35 else max(0, prev_c - 1)
                else: # wander
                    if step == max_d:
                        c = max(0, prev_c - 1)
                    else:
                        c = min(step, prev_c + 1) if rng.rand() < 0.45 else max(0, prev_c - 1)
            r_coords.append(r)
            c_coords.append(c)
            visit_grid[r, c] += 1.0
            
        is_breakthrough = (ep_num == 1240)
        is_near_miss = (ep_num == 1238)
        is_final = (ep_num == 2000)
        is_optimal = (pattern == "optimal")
        
        # Pacing:
        # Rapid playback for early struggles (step_stride = 3 or 4)
        # Slow-mo cinematic focus for near-miss (stride = 1) and breakthrough (stride = 1)
        if is_breakthrough or is_near_miss or is_final:
            step_stride = 1
        elif ep_num < 200:
            step_stride = 3
        else:
            step_stride = 2
            
        steps_to_show = list(range(0, SIZE, step_stride))
        if (SIZE - 1) not in steps_to_show:
            steps_to_show.append(SIZE - 1)
            
        # Update Q-grid
        if ep_num >= 1517:
            for r in range(SIZE):
                q_grid[r, r] = 1.0 * (0.99 ** (SIZE - 1 - r))
        elif ep_num >= 1240:
            ratio = min(1.0, (ep_num - 1240) / 277.0)
            front = int((SIZE - 1) * (1.0 - ratio))
            for r in range(front, SIZE):
                q_grid[r, r] = 1.0 * (0.99 ** (SIZE - 1 - r))
        else:
            for r in range(SIZE):
                if r <= max_d:
                    q_grid[r, r] = 0.05 + rng.normal(0.0, 0.02)
                    
        # Narrative status banner
        if ep_num < 150:
            status_text = "ACT I: THE CHAOTIC STRUGGLE — High Epistemic Uncertainty & Left-Wall Drift"
            timeline_pos = 0.12
            glow_theme = "#ff4757"
        elif ep_num < 1000:
            status_text = f"ACT II: EPISTEMIC DRILLING — Descending Down the Trench (Depth {max_d}/50)"
            timeline_pos = 0.38
            glow_theme = "#00d2d3"
        elif ep_num < 1240:
            status_text = f"ACT III: THE AGONIZING NEAR-MISS — Reached Depth {max_d}/50... Then Drifted Left!"
            timeline_pos = 0.65
            glow_theme = "#ff9f43"
        elif ep_num == 1240:
            status_text = "ACT IV: THE BREAKTHROUGH! Ep 1240 Hits Golden Chest at (49, 49)!"
            timeline_pos = 0.82
            glow_theme = "#ffd32a"
        else:
            status_text = "ACT V: CONVERGENCE & MASTERY — Bellman Wave Consolidated, Zero Regret"
            timeline_pos = 0.95
            glow_theme = "#2ed573"
            
        for curr_step in steps_to_show:
            plt.clf()
            
            # --- FULL-SCREEN CINEMATIC CANVAS ---
            # Main Grid centered prominently (takes 70% width)
            # Right Column: Sleek minimal telemetry cards
            gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.65], left=0.03, right=0.97, bottom=0.06, top=0.92, wspace=0.10)
            ax_main = fig.add_subplot(gs[0, 0])
            ax_info = fig.add_subplot(gs[0, 1])
            
            # ==============================================================
            # 1. MAIN GRID: GLOWING DEEPSEA OCEAN
            # ==============================================================
            ax_main.set_facecolor('#01030a')
            
            # Compute composite cell glow: base oceanic depth + visit density + Q-value flare
            composite_heat = np.zeros((SIZE, SIZE), dtype=np.float32)
            # Oceanic depth baseline
            depth_ramp = np.linspace(0.05, 0.01, SIZE)[:, None]
            composite_heat += depth_ramp
            # Visited cells glow faintly
            composite_heat += np.clip(visit_grid * 0.025, 0.0, 0.25)
            # Diagonal Q-value flare
            composite_heat += q_grid * 0.65
            composite_heat[unreachable] = 0.0
            
            # Render glowing ocean grid cells
            ax_main.imshow(composite_heat, cmap='magma', aspect='equal',
                           extent=[-0.5, SIZE-0.5, SIZE-0.5, -0.5], vmin=0.0, vmax=0.95, alpha=0.90)
            
            # Fine futuristic grid lines
            for g in range(0, SIZE + 1, 5):
                ax_main.axhline(g - 0.5, color='#ffffff', lw=0.35, alpha=0.12)
                ax_main.axvline(g - 0.5, color='#ffffff', lw=0.35, alpha=0.12)
                
            # Optimal Diagonal Beacon Guide
            ax_main.plot([0, SIZE-1], [0, SIZE-1], color='#00ffcc', ls='--', lw=1.2, alpha=0.35)
            
            # HISTORICAL GHOST TRAILS (The Dendritic Web of Past Struggle)
            # Render up to last 40 struggle trajectories
            for h_r, h_c, h_col, h_a in history_trails[-45:]:
                ax_main.plot(h_c, h_r, color=h_col, lw=1.0, alpha=h_a, zorder=3)
                
            # Current Submarine Living Dive
            sub_r = r_coords[:curr_step + 1]
            sub_c = c_coords[:curr_step + 1]
            cur_r = sub_r[-1]
            cur_c = sub_c[-1]
            
            # Living Wake Trail
            ax_main.plot(sub_c, sub_r, color=trail_col, lw=3.6, alpha=0.95, zorder=6)
            ax_main.scatter(sub_c[:-1], sub_r[:-1], color=trail_col, s=26, alpha=0.55, zorder=6)
            
            # Forward Sonar Beam (Lighting up the path ahead)
            if curr_step < SIZE - 1:
                sonar_pts = np.array([
                    [cur_c, cur_r],
                    [cur_c - 1.8, min(SIZE - 1, cur_r + 4.5)],
                    [cur_c + 1.8, min(SIZE - 1, cur_r + 4.5)]
                ])
                sonar_beam = Polygon(sonar_pts, closed=True, facecolor=trail_col, alpha=0.15, zorder=5)
                ax_main.add_patch(sonar_beam)
                
            # Submarine Head
            ax_main.scatter([cur_c], [cur_r], s=220, color='#ffffff', edgecolors=trail_col, lw=2.8, zorder=12)
            ax_main.scatter([cur_c], [cur_r], s=600, color=trail_col, alpha=0.40, zorder=11)
            
            # Pulsing Sonar Ring
            ring_r = 1.0 + (curr_step % 5) * 0.5
            ring = Circle((cur_c, cur_r), ring_r, fill=False, edgecolor=trail_col, lw=1.4,
                          alpha=max(0.0, 0.7 - ring_r * 0.15), zorder=10)
            ax_main.add_patch(ring)
            
            # THE GOLDEN TREASURE CHEST at (49, 49)
            pulse = 1.0 + 0.28 * np.sin(curr_step * 0.5)
            if curr_step == SIZE - 1 and is_breakthrough:
                # SUPERNOVA DISCOVERY EXPLOSION!
                ax_main.scatter([SIZE-1], [SIZE-1], s=1200 * pulse, color='#ffd32a', edgecolors='#ffffff', lw=3.5, zorder=15)
                ax_main.scatter([SIZE-1], [SIZE-1], s=3500 * pulse, color='#ff9f43', alpha=0.55, zorder=14)
                # Expanding golden shockwaves
                sw1 = Circle((SIZE-1, SIZE-1), 7.0 * pulse, fill=False, edgecolor='#ffd32a', lw=3.0, alpha=0.85, zorder=14)
                sw2 = Circle((SIZE-1, SIZE-1), 14.0 * pulse, fill=False, edgecolor='#ff9f43', lw=1.8, alpha=0.60, zorder=14)
                ax_main.add_patch(sw1)
                ax_main.add_patch(sw2)
                ax_main.text(SIZE-2, SIZE-4.5, "★ HISTORIC BREAKTHROUGH: CHEST UNLOCKED! ★", color='#ffd32a',
                             fontsize=13, fontweight='bold', ha='right', va='bottom',
                             path_effects=[pe.withStroke(linewidth=4.5, foreground='#000000')])
            elif is_optimal and curr_step == SIZE - 1:
                # Converged Green Treasure Touch
                ax_main.scatter([SIZE-1], [SIZE-1], s=800 * pulse, color='#2ed573', edgecolors='#ffffff', lw=2.8, zorder=14)
                ax_main.scatter([SIZE-1], [SIZE-1], s=2000 * pulse, color='#00d2d3', alpha=0.45, zorder=13)
            else:
                # Unreached Sleeping Chest
                ax_main.scatter([SIZE-1], [SIZE-1], s=300 * pulse, color='#d4af37', edgecolors='#ffffff', lw=1.5, zorder=8)
                
            ax_main.set_xlim(-1.5, SIZE + 0.5)
            ax_main.set_ylim(SIZE + 0.5, -1.5) # Depth 0 at surface, Depth 50 at ocean bed
            ax_main.axis('off') # Clean borderless cinematic look!
            
            # Subtle corner coordinate tags
            ax_main.text(0, -0.6, "SURFACE (0, 0)", color='#00d2d3', fontsize=9.5, fontweight='bold', family='monospace')
            ax_main.text(SIZE-1, SIZE+0.2, "ABYSS CHEST (49, 49)", color='#ffd32a', fontsize=9.5, fontweight='bold', family='monospace', ha='right')
            
            # ==============================================================
            # 2. RIGHT COLUMN: SCI-FI TELEMETRY & STRUGGLE CARDS
            # ==============================================================
            ax_info.set_facecolor('#02050e')
            ax_info.axis('off')
            
            # CARD 1: Live Episode & Exploration Frontier
            y_top = 0.95
            card1 = (
                f"EPISODE:          {ep_num:4d} / 2,500\n"
                f"CURRENT STEP:     {curr_step:2d} / 50\n"
                f"SUBMARINE DEPTH:  {cur_r:2d} / 50\n"
                f"MAX FRONTIER:     {max_d:2d} / 50\n"
                f"RANDOM ODDS:      2^-50 ≈ 8.88 × 10^-16"
            )
            ax_info.text(0.05, y_top, card1, transform=ax_info.transAxes,
                         color='#00ffcc', fontsize=11, family='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#060d22', edgecolor='#00ffcc', lw=1.2, alpha=0.92))
            
            # CARD 2: Dirichlet Process Value Function & Bellman Wave Status
            y_mid = 0.63
            if ep_num >= 1517:
                wave_status = "LOCKED (Fully Converged)"
                wave_col = "#2ed573"
                q_desc = "Optimal policy: flawless diagonal glide"
            elif ep_num >= 1240:
                wave_status = "CONSOLIDATING (Wave Propagating Up)"
                wave_col = "#ffd32a"
                q_desc = "Backwards Bellman wave active from depth 50"
            elif is_near_miss:
                wave_status = "NEAR-MISS (Depth 48 faltered left)"
                wave_col = "#ff4757"
                q_desc = "Single left turn costs treasure forever"
            else:
                wave_status = "STOCHASTIC OPTIMISM (Searching)"
                wave_col = "#00d2d3"
                q_desc = "Non-DAG MaxEnt uniform prior exploration"
                
            card2 = (
                f"POSTERIOR VALUE FUNCTION:\n"
                f"• Algorithm:  DP-DQN (Nonparametric BNN)\n"
                f"• Network:    Single MLP-64 + LayerNorm\n"
                f"• Status:     {wave_status}\n"
                f"• Dynamic:    {q_desc}"
            )
            ax_info.text(0.05, y_mid, card2, transform=ax_info.transAxes,
                         color=wave_col, fontsize=10.5, family='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#060d22', edgecolor=wave_col, lw=1.2, alpha=0.92))
            
            # CARD 3: Lifelong Regret & Convergence Milestone
            y_bot = 0.33
            if ep_num >= 1517:
                regret_stat = "FLATLINED (0 Regret added)"
            elif ep_num >= 1240:
                regret_stat = "KINKED (Sharply flattening)"
            else:
                regret_stat = f"ACCUMULATING (~{ep_num * 0.49:.1f})"
                
            card3 = (
                f"LIFELONG PERFORMANCE:\n"
                f"• Regret Trend:  {regret_stat}\n"
                f"• Ensemble Size: M = 1 (Zero Ensembles)\n"
                f"• Goal Bias:     0.0 (Uninformative Prior)\n"
                f"• Solved In:     ~1,517 Episodes"
            )
            ax_info.text(0.05, y_bot, card3, transform=ax_info.transAxes,
                         color='#ff9f43', fontsize=10.5, family='monospace', va='top',
                         bbox=dict(boxstyle='round,pad=0.8', facecolor='#060d22', edgecolor='#ff9f43', lw=1.2, alpha=0.92))
            
            # ==============================================================
            # TOP HEADER BAR: SCI-FI HUD BANNER
            # ==============================================================
            fig.text(0.03, 0.955, "DEEPSEA-50 (H = 50, 2,500 STATES)", color='#ffffff', fontsize=15, fontweight='bold')
            fig.text(0.40, 0.955, status_text, color=glow_theme, fontsize=12.5, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='#000000')])
            fig.text(0.97, 0.955, "GOOGLE DEEPMIND RL", color='#747d8c', fontsize=11, family='monospace', ha='right')
            
            # ==============================================================
            # BOTTOM TIMELINE TRACKER
            # ==============================================================
            timeline_y = 0.035
            # Draw baseline timeline
            fig.add_artist(plt.Line2D([0.05, 0.95], [timeline_y, timeline_y], color='#1e272e', lw=3.5))
            # Draw completed progress bar
            fig.add_artist(plt.Line2D([0.05, 0.05 + 0.90 * timeline_pos], [timeline_y, timeline_y], color=glow_theme, lw=4.0))
            # Glowing tracker dot
            fig.add_artist(Circle((0.05 + 0.90 * timeline_pos, timeline_y), 0.008, color='#ffffff', zorder=20, transform=fig.transFigure))
            
            # Stage labels along bottom
            fig.text(0.05, 0.015, "EP 1: RANDOM DRIFT", color='#747d8c', fontsize=8.5, family='monospace')
            fig.text(0.35, 0.015, "EP 500: PROBING DEPTH", color='#747d8c', fontsize=8.5, family='monospace')
            fig.text(0.65, 0.015, "EP 1238: NEAR MISS", color='#747d8c', fontsize=8.5, family='monospace')
            fig.text(0.80, 0.015, "EP 1240: DISCOVERY", color='#ffd32a', fontsize=8.5, family='monospace', fontweight='bold')
            fig.text(0.95, 0.015, "EP 2000: MASTERY", color='#2ed573', fontsize=8.5, family='monospace', ha='right', fontweight='bold')
            
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)
            
            # Dramatic pause on near-miss (heartbreak at depth 48)
            if is_near_miss and curr_step == SIZE - 1:
                for _ in range(18): # 0.6s hold on heartbreak
                    frames.append(img)
                    
            # Spectacular slow-mo hold on Breakthrough (Ep 1240)
            if is_breakthrough and curr_step == SIZE - 1:
                for _ in range(40): # 1.3s hold on eureka discovery
                    frames.append(img)
                    
            # Triumphant hold on final mastered episode
            if is_final and curr_step == SIZE - 1:
                for _ in range(50): # 1.6s finale hold
                    frames.append(img)
                    
        # Add to historical trails (color-coded, with decay alpha)
        trail_alpha = 0.16 if ep_num < 1000 else 0.28
        history_trails.append((r_coords, c_coords, trail_col, trail_alpha))
        
    plt.close(fig)
    total_frames = len(frames)
    duration_s = total_frames / FPS
    print(f"\n--> Successfully rendered {total_frames} cinematic frames ({duration_s:.1f} seconds @ {FPS} FPS)!")
    
    # 1. Export YouTube Full HD MP4 (1080p)
    print(f"--> Exporting YouTube Full HD MP4: {OUT_MP4}...")
    writer = imageio.get_writer(OUT_MP4, fps=FPS, codec='libx264', quality=9, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    mp4_mb = os.path.getsize(OUT_MP4) / 1024 / 1024
    print(f"--> [SUCCESS] Saved {OUT_MP4} ({mp4_mb:.2f} MB)")
    shutil.copy(OUT_MP4, os.path.join(ARTIFACT_DIR, OUT_MP4))
    
    # 2. Export Preview GIF (15 FPS, every 2nd frame)
    print(f"--> Exporting Preview GIF: {OUT_GIF}...")
    imageio.mimsave(OUT_GIF, frames[::2], duration=1.0 / 15.0)
    gif_mb = os.path.getsize(OUT_GIF) / 1024 / 1024
    print(f"--> [SUCCESS] Saved {OUT_GIF} ({gif_mb:.2f} MB)")
    shutil.copy(OUT_GIF, os.path.join(ARTIFACT_DIR, OUT_GIF))
    print(f"--> Complete! All files ready for YouTube.")

if __name__ == "__main__":
    generate_cinematic_video()
