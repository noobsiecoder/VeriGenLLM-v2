// Find the minimum of four unsigned 8-bit numbers
module top_module_cond (
    input [7:0] a, b, c, d,
    output [7:0] min
);
// Use conditional operators to build 2-way comparisons
// Combine them to create 4-way minimum