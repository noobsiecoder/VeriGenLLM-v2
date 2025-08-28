// Build a 32-bit adder-subtractor
module top_module_addsub (
    input [31:0] a,
    input [31:0] b,
    input sub,
    output [31:0] sum
);
// Use XOR gates to conditionally invert b when sub=1
// Connect sub to carry-in for two's complement
// Instantiate two add16 modules for 32-bit operation
// Note: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )
// Insert code here