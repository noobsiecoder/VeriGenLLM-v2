# This Dockerfile is for RLFT in a cloud instance (AWS/Azure/GCP)
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 21st, 2025
# Place:  Boston, MA

# Build stage with secrets
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

# Get arguments
# Secrets
ARG GCP_STORAGE_JSON_FILE
ARG MODELS_API_ENV_FILE

RUN apt-get update && apt-get install -y git curl && apt-get clean

# Work directory of the application
WORKDIR /src

# Copy contents to src/
COPY . .

# Write secrets
RUN mkdir -p secrets && \
    echo "${GCP_STORAGE_JSON_FILE}" | base64 -d > secrets/gcp-storage.json && \
    echo "${MODELS_API_ENV_FILE}" | base64 -d > secrets/models-api.env

# Image (OS) type
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Prevent interactive prompts + added Time Zone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependency tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    iverilog \
    yosys \
    && apt-get clean

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --yes
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Work directory of the application
WORKDIR /src

# Copy contents to src/
COPY . .

# Install python dependencies
# NOTE: Runs as a cautionary step
RUN uv sync

# Make script executable
RUN chmod +x scripts/cd.sh

# Runner/Executable point
CMD ["./scripts/cd.sh"]
