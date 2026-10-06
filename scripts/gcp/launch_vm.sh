#!/usr/bin/env bash
# Automated launcher for Cart-Pole Swing-Up 60-Run Benchmark on Google Cloud Compute Engine
set -e

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

INSTANCE_NAME="cartpole-benchmark-vm"
ZONE="us-central1-a"
MACHINE_TYPE="c2-standard-8"  # 8 compute-optimized vCPUs, fits perfectly within 12 CPU quota
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="50GB"
NUM_WORKERS=8

echo "=========================================================="
echo " Google Cloud 60-Run Benchmark Provisioner"
echo "=========================================================="

# Check authentication
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [ -z "$ACTIVE_ACCOUNT" ]; then
    echo "Error: No active GCP account found."
    echo "Please authenticate by running:"
    echo "    gcloud auth login"
    exit 1
fi
echo "Authenticated as: $ACTIVE_ACCOUNT"

# Check project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "Error: No GCP Project is set."
    echo "Run: gcloud projects list"
    echo "Then: gcloud config set project <PROJECT_ID>"
    exit 1
fi
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

echo "Waiting for instance to initialize SSH service (30s)..."
sleep 30

echo "Creating lightweight package of unified codebase..."
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='*.png' --exclude='*.pdf' --exclude='*.json' --exclude='results*' -czf /tmp/dp_bnns_code.tar.gz unified_dp_dqn tests

echo "Transferring package to VM..."
gcloud compute scp --quiet --project="$PROJECT_ID" --zone="$ZONE" /tmp/dp_bnns_code.tar.gz "$INSTANCE_NAME":/tmp/dp_bnns_code.tar.gz
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo mkdir -p /opt/dp_bnns && sudo tar -xzf /tmp/dp_bnns_code.tar.gz -C /opt/dp_bnns && sudo chown -R \$USER:\$USER /opt/dp_bnns"

echo "Installing prerequisites and launching $NUM_WORKERS parallel workers for 5 seeds (30 runs)..."
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo apt-get update -y && sudo apt-get install -y python3 python3-pip htop tmux && pip3 install torch numpy matplotlib && cd /opt/dp_bnns && nohup python3 -u unified_dp_dqn/run_benchmark_pool.py --env cartpole_swingup --algos dp_dqn_haar dp_dqn_non_haar boot_dqn bdqn dp_dqn_alpha_small vanilla_dqn --seeds 42 43 44 45 46 --alpha 3.0 --episodes 2500 --workers $NUM_WORKERS --out_dir ./results_cartpole_30runs > /opt/benchmark.log 2>&1 &"

echo ""
echo "=========================================================="
echo " VM Successfully Provisioned and Benchmark Launched!"
echo " Machine: $INSTANCE_NAME ($MACHINE_TYPE, 60 vCPUs)"
echo " To monitor live logs:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"tail -f /opt/benchmark.log\""
echo " To download results when done:"
echo "    gcloud compute scp --recurse $INSTANCE_NAME:/opt/dp_bnns/results_rl/cartpole_60runs ./results_rl/"
echo " To delete the VM when finished:"
echo "    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
echo "=========================================================="
