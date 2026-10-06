#!/usr/bin/env bash
# Startup script for Google Cloud Compute Engine instance
set -e

echo "=== [1/4] Updating packages and installing prerequisites ==="
apt-get update -y
apt-get install -y git python3 python3-pip python3-venv htop tmux rsync

echo "=== [2/4] Installing PyTorch and dependencies ==="
pip3 install --upgrade pip
pip3 install torch numpy matplotlib

echo "=== [3/4] Preparing benchmark environment ==="
cd /opt/dp_bnns

echo "=== [4/4] Launching 60-Run Benchmark (6 Algorithms × 10 Seeds) ==="
python3 -u src/launch_60runs_pool.py --workers 60 --episodes 2500 --out_dir ./results_rl/cartpole_60runs > /opt/benchmark.log 2>&1 &

echo "=== Benchmark launched in background. Monitoring log at /opt/benchmark.log ==="
