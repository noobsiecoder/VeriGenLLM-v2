// Build 100-bit vector reversal
module top_module_vector100 (
    input [99:0] in,
    output reg [99:0] out
);
// Reverse the bit order of the input vector:
// - in[0] should map to out[99]
// - in[1] should map to out[98]
// - in[99] should map to out[0]
// Use procedural assignment in always block
// Insert code here