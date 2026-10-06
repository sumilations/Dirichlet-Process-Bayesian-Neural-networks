#!/usr/bin/env bash
# Automated launcher for DeepSea 20 Baselines (BootDQN, BDQN, Vanilla DQN) on Google Cloud Compute Engine
set -e

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

INSTANCE_NAME="deepsea20-baselines-vm"
ZONE="us-central1-a"
MACHINE_TYPE="c2-standard-8"  # 8 compute-optimized vCPUs
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="50GB"
NUM_WORKERS=8

echo "=========================================================="
echo " Google Cloud DeepSea 20 Baselines Provisioner"
echo "=========================================================="

# Check authentication
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [ -z "$ACTIVE_ACCOUNT" ]; then
    echo "Error: No active GCP account found."
    exit 1
fi
echo "Authenticated as: $ACTIVE_ACCOUNT"

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
echo "Target GCP Project: $PROJECT_ID"

echo "Ensuring Compute Engine API is enabled..."
gcloud services enable compute.googleapis.com --project="$PROJECT_ID" || true

echo ""
echo "Creating Compute Engine VM: $INSTANCE_NAME ($MACHINE_TYPE in $ZONE)..."
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --scopes=cloud-platform \
    --quiet

echo "Waiting for instance to initialize SSH service (35s)..."
sleep 35

echo "Creating lightweight package of unified codebase..."
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='*.png' --exclude='*.pdf' --exclude='*.json' --exclude='results*' -czf /tmp/dp_bnns_code.tar.gz unified_dp_dqn src

echo "Transferring package to VM..."
gcloud compute scp --quiet --project="$PROJECT_ID" --zone="$ZONE" /tmp/dp_bnns_code.tar.gz "$INSTANCE_NAME":/tmp/dp_bnns_code.tar.gz
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo mkdir -p /opt/dp_bnns && sudo tar -xzf /tmp/dp_bnns_code.tar.gz -C /opt/dp_bnns && sudo chown -R \$USER:\$USER /opt/dp_bnns"

echo "Installing prerequisites and launching 8 parallel workers for 10 seeds (30 runs, 10,000 episodes)..."
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo apt-get update -y && sudo apt-get install -y python3 python3-pip htop tmux && pip3 install torch numpy scipy && cd /opt/dp_bnns && nohup python3 -u unified_dp_dqn/run_benchmark_pool.py --env deep_sea --size 20 --algos boot_dqn bdqn vanilla_dqn --seeds 42 43 44 45 46 47 48 49 50 51 --episodes 10000 --workers $NUM_WORKERS --out_dir ./results_deepsea20_baselines_10000ep > /opt/benchmark.log 2>&1 &"

echo ""
echo "=========================================================="
echo " VM Successfully Provisioned and Benchmark Launched on GCP!"
echo " Machine: $INSTANCE_NAME ($MACHINE_TYPE, 8 vCPUs)"
echo " Algorithms: BootDQN + Rand Priors, BDQN, Vanilla DQN"
echo " Seeds: 42 to 51 (10 seeds, 30 runs total)"
echo " Episodes: 10,000 per run"
echo "=========================================================="
