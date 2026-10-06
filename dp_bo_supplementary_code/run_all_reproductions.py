#!/usr/bin/env python3
"""
Master Reproduction Script for Supplementary Material.
Executes and displays all tables and figures from the ICLR submission:
- Table 1: 4-Benchmark Multimodal Suite (Ackley 2D, Levy 2D, Rosenbrock 4D, Rastrigin 4D)
- Table D.1: 60-Dimensional Rover Trajectory Planning
- Table E.1: 32-Dimensional Continuous Control (Pendulum-v1)
- Figure 1: Multimodal Showdown (figure_dp_bo_showdown.png)
- Figure D.1: 60D Rover Convergence & Navigation Map (figure_rover_60d.png)
"""

import os
import sys
import subprocess

def run_script(script_path, description):
    print("\n" + "=" * 80)
    print(f"[*] {description}")
    print("=" * 80)
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    if res.returncode != 0:
        print(f"[!] Warning: {script_path} exited with code {res.returncode}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")

    print("=" * 80)
    print("Data-Space Dirichlet Process Bayesian Neural Networks (DP-BNNs)")
    print("Official Supplementary Material Reproduction Suite")
    print("=" * 80)

    # 1. Table 1
    run_script(os.path.join(scripts_dir, "run_table1_multimodal.py"), "Reproducing Table 1: Multimodal Benchmarks")

    # 2. Table D.1
    run_script(os.path.join(scripts_dir, "run_rover_60d.py"), "Reproducing Table D.1: 60D Rover Trajectory Planning")

    # 3. Table E.1
    run_script(os.path.join(scripts_dir, "run_pendulum_32d.py"), "Reproducing Table E.1: 32D Pendulum Continuous Control")

    # 4. Figure 1
    run_script(os.path.join(scripts_dir, "plot_figure1_showdown.py"), "Regenerating Figure 1 (figure_dp_bo_showdown.png)")

    # 5. Figure D.1
    run_script(os.path.join(scripts_dir, "plot_figure_rover_60d.py"), "Regenerating Figure D.1 (figure_rover_60d.png)")

    print("\n" + "=" * 80)
    print("[+] All reproductions completed successfully!")
    print(f"[+] Output figures saved in: {os.path.join(base_dir, 'figures')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
