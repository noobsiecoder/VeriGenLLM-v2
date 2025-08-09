# test_wire_assign_testbench.py
# Problem: example_02
"""
Test suite for wire assignment Verilog code generation

This test file validates that the LLM correctly generates a simple
wire assignment statement (assign a = 1'b1) and verifies both the
code correctness and actual execution through the testbench.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 4th, 2025

Note: LLM was used to generate comments
"""

import os
import subprocess
import tempfile
from dataset.test.example_02.testbench import check_code


def test_check_code_with_proper_code():
    """
    Test that correct wire assignment code passes validation

    The expected solution is a simple assignment of logic high (1'b1)
    to wire 'a'. This tests the basic case where the LLM generates
    exactly what's expected.
    """
    code = """
    assign a = 1'b1;
    """
    assert check_code(code)


def test_real_scenario():
    """
    Test end-to-end execution of wire assignment code

    This test simulates the actual evaluation pipeline:
    1. Creates a temporary Verilog file with the code
    2. Runs the testbench.py script against it
    3. Verifies successful execution (returncode == 0)

    This ensures the code not only passes validation but also
    works correctly when executed through the full pipeline.
    """
    code = """
    assign a = 1'b1;
    """
    # First verify the code passes basic validation
    assert check_code(code)

    # Create temporary Verilog file
    temp_sol_filepath = None
    with tempfile.NamedTemporaryFile(suffix=".v", mode="w+", delete=False) as temp_file:
        temp_file.write(code)
        temp_sol_filepath = temp_file.name

    # Run the actual testbench script as the evaluation pipeline would
    test_py_path = "dataset/test/example_02/testbench.py"
    result = subprocess.run(
        ["uv", "run", test_py_path, temp_sol_filepath],
        capture_output=True,
        text=True,
    )

    # Clean up temporary file
    os.remove(temp_sol_filepath)

    # Verify successful execution
    assert result.returncode == 0


def test_check_code_with_incorrect_code():
    """
    Test that incorrect wire assignments fail validation

    Tests two common mistakes:
    1. Wrong bit width specification (0'b1 instead of 1'b1)
    2. Missing bit width specification (just '1' instead of 1'b1)

    These tests ensure the validation is strict about proper
    Verilog syntax for constant assignments.
    """
    # Test 1: Incorrect bit width (0 bits for value 1)
    code = """
    assign a = 0'b1;
    """
    assert not check_code(code)

    # Test 2: Missing bit width specification
    code = """
    assign a = 1;
    """
    assert not check_code(code)
