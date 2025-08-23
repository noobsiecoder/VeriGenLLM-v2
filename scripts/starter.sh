#!/bin/bash
# Base script for CD in Azure/GCP VM:
#   1) Checks Docker (installs if missing)
#   2) Ensures NVIDIA drivers for GPU are installed
#   3) Builds & runs Docker image for RLFT
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 21st, 2025
# Place:  Boston, MA

set -e
GITHUB_REPO_URI=$1
GITHUB_REPO_BRANCH=$2
GCP_SECRETS_FILE=$3
APIKEYS_FILE=$4
REPO_NAME="VeriGenLLM-v2"
DOCKER_IMAGE_NAME="verilog-rlft"
LOGFILE="/var/log/rlft_setup.log"
BUILD_LOGFILE="/var/log/docker_build.log"
RUN_LOGFILE="/var/log/docker_run.log"

# Step 0: Redirect all logs
exec > >(sudo tee -a "$LOGFILE") 2>&1
echo "===== Starting RLFT setup at $(date) ====="

# Step 1: Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    sudo apt-get update -y
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# Step 2: Ensure NVIDIA drivers for GPU support
# NOTE: To install NVIDIA Drivers in Standard NVadsA10_v5 VM (Ubuntu 22.04), 
# Use the following link:
# https://forums.developer.nvidia.com/t/installing-nvidia-drivers-cuda-on-azure-nvadsa10-v5-vm-ubuntu-22-04/321128/3
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Exiting..."
    exit 0
else
    echo "NVIDIA drivers already installed."
fi

# Step 3: Ensure git is installed
if ! command -v git &> /dev/null; then
    echo "git not found."
    sudo apt-get update -y
    sudo apt-get install -y git-all
else
    echo "git already installed."
fi

# Step 3: Make project directory
mkdir -p src/
cd src/

# Clone repo if not already present
if [ ! -d "$REPO_NAME" ]; then
  git clone -b enhance-v1 $GITHUB_REPO_URI.git $REPO_NAME
  echo "Cloned Repository"

  cd $REPO_NAME
  echo "In repo directory"
fi

# Step 4: Build Docker image
echo "Building Docker image: $DOCKER_IMAGE_NAME"
sudo docker build -f Dockerfile \
    --build-arg REPO_URL=$GITHUB_REPO_URI \
    --build-arg BRANCH_NAME=$GITHUB_REPO_BRANCH \
    --build-arg GCP_STORAGE_JSON_FILE=$GCP_SECRETS_FILE \
    --build-arg MODELS_API_ENV_FILE=$APIKEYS_FILE \
    --no-cache \
    -t $DOCKER_IMAGE_NAME . \
    | tee -a "$BUILD_LOGFILE"

# Step 5: Run container with GPU (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Running container with GPU support..."
    sudo docker run --gpus all $DOCKER_IMAGE_NAME \
        | tee -a "$RUN_LOGFILE"
else
    echo "No GPU detected. Running container without GPU..."
    sudo docker run $DOCKER_IMAGE_NAME \
        | tee -a "$RUN_LOGFILE"
fi

echo "===== Finished RLFT setup at $(date) ====="
