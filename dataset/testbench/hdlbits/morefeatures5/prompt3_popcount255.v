// More verilog features
module top_module_popcount255 ( 
    input [254:0] in,
    output reg [7:0] out
);
// A "population count" circuit counts the number of '1's in an input vector. Build a population count circuit for a 255-bit input vector.
// Hint: Use a for loop to find out the number of 1's
// Requirements:
// 1. Count total number of '1' bits in 255-bit input
// 2. Use for loop with procedural assignment
// 3. Output is 8-bit (max count = 255)
//
// Algorithm:
// - Reset output to 0 at start of always block
// - Iterate through all 255 bits (index 0 to 254)
// - Check each bit: if it's 1, increment counter
// - Final out value = total number of 1's
//
// Alternative implementations:
// 1. Simpler: out = out + in[i]; (since in[i] is 0 or 1)
// 2. Tree reduction: Divide and conquer approach
// 3. Look-up tables for smaller chunks
//
// Examples:
// - in = 255'h0 → out = 8'd0 (no ones)
// - in = 255'h7FF...FFF → out = 8'd255 (all ones)
// - in = 255'h1 → out = 8'd1 (single one)
// - in = 255'hFF → out = 8'd8 (eight ones)
//
// Note: 
// - Output must be reg for procedural assignment
// - Always use blocking assignment (=) in combinational always
// - Input is [254:0], which is 255 bits total
// Insert code here