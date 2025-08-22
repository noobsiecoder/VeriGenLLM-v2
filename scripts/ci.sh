#!/bin/bash
# This file is a CI script and aids in:
#   1) Code lint checking
#   2) Runs tests from tests/
#   3) Runs tests in Verilog dataset
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 20th, 2025
# Place:  Boston, MA
set -e

# Step 1: Code lint checking with ruff
echo "=== Running Code Linting ==="
uv run ruff check .
uv run ruff format --check .

# Step 2: Run tests
echo "=== Running Python Script(s) Tests ==="
uv run pytest -v tests/test_evals.py tests/test_models.py::test_api_connection_with_no_env_load

# Step 3: Run tests for ground truth with corresponding testbench
echo "=== Running Verilog Tests ==="
cd dataset/testbench/hdlbits
find . -name "tb_*.v" | while read tb_path; do
    dir=$(dirname "$tb_path")
    tb_file=$(basename "$tb_path")
    base=${tb_file#tb_}
    base=${base%.v}
    answer_file="answer_${base}.v"
    answer_path="${dir}/${answer_file}"
    if [ -f "$answer_path" ]; then
        echo "Testing: ${dir}/${base}"
        cd "$dir"
        iverilog -o test_${base} ${tb_file} ${answer_file}
        vvp test_${base} || (echo "✗ Test failed: ${dir}/${base}" && exit 1)
        cd - > /dev/null
    else
        echo "⚠ Missing answer file: $answer_path"
    fi
done
echo "✓ All tests passed!"
