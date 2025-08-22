#!/bin/bash
# This file is the orchestrator of workflow to ship it to cloud
# Functionality:
#   1) Checks availability for resource in each cloud services.
#   2) Deploy in a VM that is available; else, schedule-for-later/exit.
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 21st, 2025
# Place:  Boston, MA
set -e

# ==================== AWS credentials =====================
# TODO: AWS Credentials not used as quota wasn't increased yet (Recorded on: Aug 22nd, 2025)

# ==================== Azure credentials ===================
AZURE_USERNAME="$secrets.AZURE_USERNAME"
AZURE_PASSWORD="$secrets.AZURE_APP_ID"
AZURE_TENANT="$secrets.AZURE_TENANT"
AZURE_RESOURCE_GROUP="$secrets.AZURE_RESOURCE_GROUP"
AZURE_VM_INSTANCE="$secrets.AZURE_VM_INSTANCE"

# =================== GCP credentials =====================
GCP_TYPE="$secrets.GCP_TYPE"
GCP_PRIVATE_KEY_ID="$secrets.GCP_PRIVATE_KEY_ID"
GCP_PROJECT_ID="$secrets.GCP_PROJECT_ID"
GCP_PRIVATE_KEY="$secrets.GCP_PRIVATE_KEY"
GCP_CLIENT_EMAIL="$secrets.GCP_CLIENT_EMAIL"
GCP_CLIENT_ID="$secrets.GCP_CLIENT_ID"
GCP_AUTH_URI="$secrets.GCP_AUTH_URI"
GCP_TOKEN_URI="$secrets.GCP_TOKEN_URI"
GCP_CERT="$secrets.GCP_CERT"
GCP_CERT_URI="$secrets.GCP_CERT_URI"
GCP_DOMAIN="$secrets.GCP_DOMAIN"

FILE_ENTRYPOINT="~/VeriGenLLM-v2/main.py"

# TODO: Yet to work on AWS VMs -> Waiting on quota increase
# Function to check AWS VM


# Function to check AZURE VM
# TODO: Instead of the pytho script, check dockerfile
check_azure() {
    # Checking Azure container
    echo "Checking Azure VM for running Docker containers..."

    # Login to Azure
    az login --service-principal -u $AZURE_USERNAME -p $AZURE_PASSWORD --tenant $AZURE_TENANT > /dev/null 2>&1

    # Check if any Docker containers are running (excluding system containers)
    # This command will:
    # 1. List all running containers with their names and images
    # 2. Exclude the Docker daemon and system containers
    # 3. Look for actual application containers
    local check_script='
        # Get list of running containers
        running_containers=$(docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null | tail -n +2)
        
        if [ -z "$running_containers" ]; then
            echo "NO_CONTAINERS"
            exit 0
        fi
        
        # Check if any non-system containers are running
        # Exclude common system containers and daemons
        app_containers=$(echo "$running_containers" | grep -v -E "^(portainer|watchtower|traefik|nginx-proxy|docker-proxy|registry)" | grep -v -E "(daemon|system)")
        
        if [ -z "$app_containers" ]; then
            echo "NO_APP_CONTAINERS"
            exit 0
        fi
        
        # Check specifically for VeriGenLLM-v2 related containers
        verigen_containers=$(echo "$app_containers" | grep -i "verigen\|llm\|rlft")
        
        if [ ! -z "$verigen_containers" ]; then
            echo "VERIGEN_RUNNING"
            echo "$verigen_containers"
            exit 0
        fi
        
        # Other app containers are running
        echo "OTHER_APPS_RUNNING"
        echo "$app_containers"
        exit 0
    '
    
    # Execute the check script on Azure VM
    local OUTPUT=$(az vm run-command invoke \
        -g $AZURE_RESOURCE_GROUP \
        -n $AZURE_VM_INSTANCE \
        --command-id RunShellScript \
        --scripts "$check_script" \
        --output json 2>&1)

    # Check if the command executed successfully
    if [ $? -ne 0 ]; then
        echo "✗ Failed to execute command on Azure VM"
        echo "Error: $OUTPUT"
        return 3  # VM unreachable or command failed
    fi

    # Parse the output
    local stdout_content=$(echo "$OUTPUT" | jq -r '.value[0].message' 2>/dev/null | grep -oP '\[stdout\]\K.*' | sed 's/\\n/\n/g')
    
    # Determine the status based on output
    if echo "$stdout_content" | grep -q "NO_CONTAINERS"; then
        echo "✓ No Docker containers running on Azure VM - VM is available"
        return 1  # VM available for deployment
    elif echo "$stdout_content" | grep -q "NO_APP_CONTAINERS"; then
        echo "✓ No application containers running on Azure VM - VM is available"
        return 1  # VM available for deployment
    elif echo "$stdout_content" | grep -q "VERIGEN_RUNNING"; then
        echo "✗ VeriGenLLM-v2 is already running on Azure VM"
        # Extract and display the container details
        local container_info=$(echo "$stdout_content" | grep -A 10 "VERIGEN_RUNNING" | tail -n +2 | head -n -1)
        echo "Running containers:"
        echo "$container_info"
        return 0  # VM busy with our application
    elif echo "$stdout_content" | grep -q "OTHER_APPS_RUNNING"; then
        echo "⚠ Other applications are running on Azure VM"
        # Extract and display the container details
        local container_info=$(echo "$stdout_content" | grep -A 10 "OTHER_APPS_RUNNING" | tail -n +2 | head -n -1)
        echo "Running containers:"
        echo "$container_info"
        return 0  # VM busy with other applications
    else
        echo "✗ Unable to determine Azure VM status"
        return 3  # Unknown status
    fi
}

# Function to check GCP VM
# TODO: Instead of the pytho script, check dockerfile
check_gcp() {
    # Replace literal '\n' with actual newlines in private key
    local FIXED_PRIVATE_KEY
    FIXED_PRIVATE_KEY=$(echo "$GCP_PRIVATE_KEY" | sed 's/\\n/\n/g')

    # Write security object to /tmp/gcp-secret.json
    cat > /tmp/gcp-secret.json <<EOF
        {
        "type": "$GCP_TYPE",
        "project_id": "$GCP_PROJECT_ID",
        "private_key_id": "$GCP_PRIVATE_KEY_ID",
        "private_key": "$FIXED_PRIVATE_KEY",
        "client_email": "$GCP_CLIENT_EMAIL",
        "client_id": "$GCP_CLIENT_ID",
        "auth_uri": "$GCP_AUTH_URI",
        "token_uri": "$GCP_TOKEN_URI",
        "auth_provider_x509_cert_url": "$GCP_CERT",
        "client_x509_cert_url": "$GCP_CERT_URI",
        "universe_domain": "$GCP_DOMAIN"
        }
EOF

    export GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcp-secret.json"

    # Run command on VM
    local output=$(gcloud compute ssh $GCP_INSTANCE_NAME \
        --zone=$GCP_INSTANCE_ZONE \
        --command="pgrep -af '$FILE_ENTRYPOINT'" \
        --ssh-flag="-o ConnectTimeout=10" \
        2>&1)
    
    if [ $? -eq 0 ] && [ ! -z "$output" ]; then
        echo "✓ Script is running"
        echo "Process info: $output"
        return 0
    else
        echo "✗ Script is not running"
        return 1
    fi
}

# Main Runner
main() {}

main # Run main function
