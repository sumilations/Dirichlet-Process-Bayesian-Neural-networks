#!/usr/bin/env bash
# Automated launcher for Deep Sea Scaling Benchmark with Model-Free Uniform Maximum-Entropy Base Measure
set -e

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

INSTANCE_NAME="deepsea-nondag-scaling"
ZONE="us-central1-a"
MACHINE_TYPE="c2-standard-8"  # 8 compute-optimized vCPUs (3.8 GHz turbo)
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="40GB"
NUM_WORKERS=8

echo "=========================================================="
echo " Google Cloud Deep Sea Scaling: Model-Free Non-DAG MaxEnt"
echo " Base Measure: deep_sea_nondag_maxent (Zero DAG bias, uniform optimistic)"
echo " Grid: 30 Sizes in [10, 50] x 2 modes (Pure TS & Multi-Sample) x 3 seeds = 180 runs"
echo "=========================================================="

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "Error: No GCP Project is set."
    exit 1
fi
echo "Target GCP Project: $PROJECT_ID"

echo "Ensuring Compute Engine API is enabled..."
gcloud services enable compute.googleapis.com --project="$PROJECT_ID" || true

EXISTING=$(gcloud compute instances list --filter="name=($INSTANCE_NAME)" --format="value(name)" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Found existing instance $INSTANCE_NAME. Deleting it first to ensure clean state..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
fi

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

echo "Creating package of scaling benchmark..."
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='*.png' --exclude='*.pdf' --exclude='results*' -czf /tmp/dp_nondag_scaling_pkg.tar.gz src run_scaling_sampler_comparison.py plot_scaling_sampler_comparison.py

echo "Transferring package to VM..."
gcloud compute scp --quiet --project="$PROJECT_ID" --zone="$ZONE" /tmp/dp_nondag_scaling_pkg.tar.gz "$INSTANCE_NAME":/tmp/dp_nondag_scaling_pkg.tar.gz

echo "Setting up environment and launching $NUM_WORKERS parallel workers..."
gcloud compute ssh --quiet "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command="sudo apt-get update -y && sudo apt-get install -y python3 python3-pip htop tmux && pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install numpy matplotlib bsuite && sudo mkdir -p /opt/benchmark && sudo tar -xzf /tmp/dp_nondag_scaling_pkg.tar.gz -C /opt/benchmark && sudo chown -R \$USER:\$USER /opt/benchmark && cd /opt/benchmark && nohup python3 -u run_scaling_sampler_comparison.py --base_measure deep_sea_nondag_maxent --sizes 10 11 13 14 16 17 18 20 21 22 24 25 27 28 29 31 32 33 35 36 38 39 40 42 43 44 46 47 49 50 --seeds 42 43 44 --workers $NUM_WORKERS --out_dir ./results_scaling_nondag_maxent > /opt/benchmark/scaling_nondag.log 2>&1 &"

echo ""
echo "=========================================================="
echo " VM Successfully Provisioned and Benchmark Launched!"
echo " Machine: $INSTANCE_NAME ($MACHINE_TYPE, 8 vCPUs)"
echo " Base Measure: deep_sea_nondag_maxent"
echo " To monitor live logs:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"tail -f /opt/benchmark/scaling_nondag.log\""
echo " To download results when done:"
echo "    gcloud compute scp --recurse $INSTANCE_NAME:/opt/benchmark/results_scaling_nondag_maxent ./results_scaling_nondag_maxent"
echo " To delete the VM when finished:"
echo "    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
echo "=========================================================="
