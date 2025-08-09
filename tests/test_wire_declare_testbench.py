# test_wire_declare_testbench.py
# Problem: example_01
"""
Test suite for simple wire declaration Verilog code generation

This test file validates that the LLM correctly generates a simple
wire declaration (wire a;) without any additional complexity or
extra declarations.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 4th, 2025

Note: LLM was used to generate comments
"""

from dataset.test.example_01.testbench import check_code


def test_check_code_with_proper_code():
    """
    Test that correct wire declaration passes validation

    The expected solution is a single wire declaration for 'a'.
    This is the most basic Verilog construct - declaring a wire
    without any bit width specification (defaults to 1 bit).
    """
    code = """
    wire a;
    """
    assert check_code(code)


def test_check_code_with_incorrect_code():
    """
    Test that incorrect wire declarations fail validation

    Tests two common mistakes:
    1. Declaring extra wires (wire b) when only 'a' is expected
    2. Wrapping the declaration in a module with wrong wire names

    These tests ensure the LLM generates exactly what's asked for,
    without adding unnecessary complexity or incorrect elements.
    """
    # Test 1: Extra wire declaration
    # The problem asks for only 'wire a', not additional wires
    code = """
    wire a;
    wire b;
    """
    assert not check_code(code)

    # Test 2: Module wrapper with incorrect wire names
    # Tests that module context and wrong wire names fail
    code = """
    module assignment;
        wire b;      // Wrong: should be 'a'
        wire a_b;    // Wrong: not requested
    endmodule
    """
    assert not check_code(code)
