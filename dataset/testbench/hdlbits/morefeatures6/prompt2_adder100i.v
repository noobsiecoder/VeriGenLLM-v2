// More verilog features
module top_module_adder100i (
    input [99:0] a, b,
    input cin,
    output [99:0] cout,
    output [99:0] sum
);
// Create a ripple-carry adder:
// - Use generate block to instantiate 100 full adders
// - First adder: takes cin as carry input
// - Other adders: take cout from previous adder
// - Each adder computes: sum = a XOR b XOR cin
// - Each adder computes: cout = (a AND b) OR (a AND cin) OR (b AND cin)
// Define the full adder module separately
// Insert code here