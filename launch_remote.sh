#!/usr/bin/env bash
set -e

mkdir -p /opt/benchmark/option_c_scaling
tar -xzf /tmp/dp_scaling_option_c.tar.gz -C /opt/benchmark/option_c_scaling
cd /opt/benchmark/option_c_scaling

# Kill any existing scaling run if running
pkill -f run_gcp_scaling_30sizes || true

# Launch the 30-size benchmark with DAG prior in background
nohup python3 -u run_gcp_scaling_30sizes.py \
    --variant dag \
    --workers 8 \
    --out_dir ./results_30sizes_dag \
    > ./scaling_dag_30sizes.log 2>&1 &

NEW_PID=$!
echo "SUCCESSFULLY_LAUNCHED_PID=$NEW_PID"
sleep 2
ps aux | grep run_gcp_scaling_30sizes | grep -v grep
