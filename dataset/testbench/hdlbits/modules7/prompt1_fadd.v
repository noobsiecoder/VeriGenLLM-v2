// Create a hierarchical 32-bit adder
module top_module_fadd (
    input [31:0] a,
    input [31:0] b
// Build using add16 modules
// Also create add1 full adder module
// Note: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )
// Insert code here