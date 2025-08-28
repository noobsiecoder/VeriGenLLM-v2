// Build 100-bit vector reversal using procedural assignment
module top_module_vector100 (
    input [99:0] in,
    output reg [99:0] out
);
// Requirements:
// 1. Reverse the entire 100-bit vector
// 2. Use procedural assignment (always block)
//
// Implementation approach:
// always @(*) begin
//     for (i = 0; i < 100; i = i + 1) begin
//         out[99 - i] = in[i];
//     end
// end
//
// Bit mapping:
// - in[0] → out[99]
// - in[1] → out[98]
// - in[2] → out[97]
// - ...
// - in[98] → out[1]
// - in[99] → out[0]
//
// Alternative implementations:
// 1. Using generate blocks (structural)
// 2. Using concatenation: out = {in[0], in[1], ..., in[99]}
// 3. Using another for loop: out[i] = in[99-i]
//
// Examples:
// - in = 100'h1 → out = 100'h1 << 99
// - in = 100'hF → out = 100'hF << 96
// - in = {50'h0, 50'hF...F} → out = {50'hF...F, 50'h0}
//
// Note: output must be declared as 'reg' for procedural assignment
// Insert code here