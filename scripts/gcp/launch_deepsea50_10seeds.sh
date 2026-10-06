#!/usr/bin/env bash
# Automated launcher for DeepSea N=50 Pure TS 10-Seed Benchmark on Google Cloud
set -e

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

INSTANCE_NAME="deepsea50-pure-ts-vm"
ZONE="us-central1-a"
MACHINE_TYPE="c2-standard-8"  # 8 compute-optimized vCPUs, well within 12 CPU quota
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="40GB"
NUM_WORKERS=7

echo "=========================================================="
echo " Google Cloud DeepSea N=50 Pure TS 10-Seed Benchmark"
echo " Comparing Scaled K_prior vs Standard K_prior (20 runs)"
echo " Up to 20,000 episodes with cumulative regret tracking"
echo "=========================================================="

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "Error: No GCP Project is set."
    exit 1
fi
echo "Target GCP Project: $PROJECT_ID"

# Check if instance already exists
EXISTING=$(gcloud compute instances list --filter="name=($INSTANCE_NAME)" --format="value(name)" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Warning: Instance $INSTANCE_NAME already exists. Deleting first to ensure a clean run..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
fi

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

echo "Creating package of benchmark..."
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='*.png' --exclude='*.pdf' --exclude='results*' -czf /tmp/deepsea50_pkg.tar.gz src run_deepsea50_pure_ts_10seeds.py

echo "Transferring package to VM..."
gcloud compute scp --quiet --project="$PROJECT_ID" --zone="$ZONE" /tmp/deepsea50_pkg.tar.gz "$INSTANCE_NAME":/tmp/deepsea50_pkg.tar.gz

echo "Setting up environment and launching $NUM_WORKERS parallel workers..."
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo apt-get update -y && sudo apt-get install -y python3 python3-pip htop tmux && pip3 install 'setuptools<70' && pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install numpy matplotlib bsuite && sudo mkdir -p /opt/benchmark && sudo tar -xzf /tmp/deepsea50_pkg.tar.gz -C /opt/benchmark && sudo chown -R \$USER:\$USER /opt/benchmark && cd /opt/benchmark && nohup python3 -u run_deepsea50_pure_ts_10seeds.py --workers $NUM_WORKERS --max_episodes 20000 --out_dir ./results_deepsea50_pure_ts_10seeds > /opt/benchmark/deepsea50.log 2>&1 &"

echo ""
echo "=========================================================="
echo " VM Successfully Provisioned and DeepSea N=50 Benchmark Launched!"
echo " Machine: $INSTANCE_NAME ($MACHINE_TYPE, 8 vCPUs)"
echo " To monitor live logs:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"tail -f /opt/benchmark/deepsea50.log\""
echo "=========================================================="
