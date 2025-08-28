// More verilog features
module top_module_bcadd100 (
    input [399:0] a, b,
    input cin,
    output cout,
    output [399:0] sum
);
// Instantiate 100 BCD adders and perform - adder, which should add two 100-digit BCD numbers (packed into 400-bit vectors)
// Hint: `bcd_fadd` already provided in this question
// Insert code here