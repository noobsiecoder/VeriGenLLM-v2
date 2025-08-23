#!/bin/bash
# This script does cloud deployment and performs critical operations:
#   1) Check Azure VM health and deploy if healthy (available)
#   2) 
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 20th, 2025
# Place:  Boston, MA
set -e

mkdir -p ~/logs
LOGFILE="rlft_training_$(date '+%F_%X').log"

log() {
    msg=$1
    echo "$(date '+%F_%X'): $msg" >> ~/logs/$LOGFILE
}

# Step 1: Run few more tests to check if LLM loads in cloud instance
log "=== Running LLM Sepecific Tests ==="
uv run pytest -v \
    tests/test_llm_prompts.py::test_prompt_extraction \
    tests/test_models.py 2>&1 | while IFS= read -r line; do
        log "$line"
    done

if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    log "✓ Pre-tests PASSED"
else
    log "✗ Pre-tests FAILED"
    exit 1
fi

# Step 2: Run RLFT
# TODO: insert command(s) here ...
log "=== Running RLFT ==="
