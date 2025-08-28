// Build a 32-bit adder using two 16-bit adder modules
module top_module_add (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
// Instantiate two add16 modules:
// - First add16: adds lower 16 bits
// - Second add16: adds upper 16 bits with carry from first
// Note: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )
// Set cin of first adder to 0
// Insert code here