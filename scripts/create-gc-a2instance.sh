#!/bin/sh

# Check if GCloud is initialized in local machine
# run `gcloud init`

# Startup file
# Create startup script file
cat > startup.sh << 'EOF'
#!/bin/bash
apt-get update
apt-get install -y curl git-all python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
pip3 install torch transformers accelerate
EOF

# Create A2 GPU Instance (a2-highgpu-1g)
for zone in us-central1-a us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b us-west4-a us-west4-b us-east1-c us-east1-d us-east4-a us-east4-c us-west3-b us-west4-b us-east7-b asia-northeast1-a asia-northeast1-c asia-northeast3-a asia-northeast3-b asia-southeast1-b asia-southeast1-c europe-west4-a europe-west4-b me-west1-a me-west1-c; do
  echo "Trying zone: $zone"
  gcloud compute instances create verilog-eval \
    --zone=$zone \
    --machine-type=g2-standard-16 \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --quiet 2>/dev/null && echo "SUCCESS in $zone" && break || echo "Failed in $zone"
done

# Connecting via SSH
# gcloud compute ssh verilog-eval --zone=europe-west4-a
