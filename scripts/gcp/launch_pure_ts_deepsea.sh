#!/usr/bin/env bash
# Automated launcher for Pure TS Scaling Enhancement (Standard vs Scaled K_prior on N=40,50)
set -e

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

INSTANCE_NAME="pure-ts-scaling-vm"
ZONE="us-central1-a"
MACHINE_TYPE="c2-standard-8"  # 8 compute-optimized vCPUs, fits comfortably within 12 CPU quota
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="40GB"
NUM_WORKERS=7

echo "=========================================================="
echo " Google Cloud Pure TS Deep Sea Scaling Enhancement"
echo " Standard K_prior (50) vs Scaled K_prior (4*N) on N=40, 50"
echo "=========================================================="

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "Error: No GCP Project is set."
    exit 1
fi
echo "Target GCP Project: $PROJECT_ID"

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
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='*.png' --exclude='*.pdf' --exclude='results*' -czf /tmp/pure_ts_pkg.tar.gz src run_pure_ts_deepsea_scaling.py

echo "Transferring package to VM..."
gcloud compute scp --quiet --project="$PROJECT_ID" --zone="$ZONE" /tmp/pure_ts_pkg.tar.gz "$INSTANCE_NAME":/tmp/pure_ts_pkg.tar.gz

echo "Setting up environment and launching $NUM_WORKERS parallel workers..."
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo apt-get update -y && sudo apt-get install -y python3 python3-pip htop tmux && pip3 install 'setuptools<70' && pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install numpy matplotlib bsuite && sudo mkdir -p /opt/benchmark && sudo tar -xzf /tmp/pure_ts_pkg.tar.gz -C /opt/benchmark && sudo chown -R \$USER:\$USER /opt/benchmark && cd /opt/benchmark && nohup python3 -u run_pure_ts_deepsea_scaling.py --workers $NUM_WORKERS --out_dir ./results_pure_ts_deepsea > /opt/benchmark/pure_ts.log 2>&1 &"

echo ""
echo "=========================================================="
echo " VM Successfully Provisioned and Pure TS Suite Launched!"
echo " Machine: $INSTANCE_NAME ($MACHINE_TYPE, 8 vCPUs)"
echo " To monitor live logs:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"tail -f /opt/benchmark/pure_ts.log\""
echo "=========================================================="
