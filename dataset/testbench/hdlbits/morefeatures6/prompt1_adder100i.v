// More verilog features
module top_module_adder100i (
    input [99:0] a, b,
    input cin,
    output [99:0] cout,
    output [99:0] sum
);
// Create a 100-bit binary ripple-carry adder by instantiating 100 full adders. The adder adds two 100-bit numbers and a carry-in to produce a 100-bit sum and carry out.
// Insert code here