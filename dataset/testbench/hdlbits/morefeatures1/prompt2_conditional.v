// Build a 4-way minimum circuit using conditional operators
module top_module_cond (
    input [7:0] a, b, c, d,
    output [7:0] min
);
// Create 2-way min circuits: min = (a < b) ? a : b
// Use intermediate wires to combine results
// Build a tree structure to find minimum of all four
// Unsigned comparison operators work normally (a < b)