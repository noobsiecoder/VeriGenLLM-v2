#!/bin/bash
# Script checks if there is a docker instance running in a cloud VM.
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 22nd, 2025
# Place:  Boston, MA
set -e

# Get list of running containers
running_containers=$(docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null | tail -n +2)

# Check output
if [ -z "$running_containers" ]; then
    echo "NO_CONTAINERS"
    exit 0
fi

# Check if any non-system containers are running
# Exclude common system containers and daemons
app_containers=$(echo "$running_containers" | grep -v -E "^(portainer|watchtower|traefik|nginx-proxy|docker-proxy|registry)" | grep -v -E "(daemon|system)")

# Check output
if [ -z "$app_containers" ]; then
    echo "NO_APP_CONTAINERS"
    exit 0
fi

# Check specifically for VeriGenLLM-v2 related containers
verigen_containers=$(echo "$app_containers" | grep -i "verigen\|llm\|rlft")

# Check output
if [ ! -z "$verigen_containers" ]; then
    echo "VERIGEN_RUNNING"
    echo "$verigen_containers"
    exit 0
fi

# Other app containers are running
echo "OTHER_APPS_RUNNING"
echo "$app_containers"
exit 0