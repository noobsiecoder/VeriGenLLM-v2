// Build a 4-way minimum finder using conditional operators
module top_module_cond (
    input [7:0] a, b, c, d,
    output [7:0] min
);
// Requirements:
// 1. Find minimum of four 8-bit unsigned numbers
// 2. Use conditional operator: (condition) ? true_value : false_value
// 3. Build using 2-way comparisons first
//
// Approach:
// 1. Create two 2-way minimums:
//    wire [7:0] min1 = (a < b) ? a : b;  // min of a,b
//    wire [7:0] min2 = (c < d) ? c : d;  // min of c,d
// 2. Find minimum of the two results:
//    wire [7:0] min3 = (min1 < min2) ? min1 : min2;
// 3. Assign to output:
//    assign min = min3;
//
// Alternative approach (sequential):
// wire [7:0] min1 = (a < b) ? a : b;
// wire [7:0] min2 = (min1 < c) ? min1 : c;
// assign min = (min2 < d) ? min2 : d;
//
// Note: Tree structure is preferred for better parallelism