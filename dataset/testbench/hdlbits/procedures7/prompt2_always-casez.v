// Build a 4-bit priority encoder using casez
module top_module_always_case (
    input [3:0] in,
    output reg [1:0] pos
);
// Priority: bit 3 (highest) to bit 0 (lowest)
// Output encoding:
// - in[3] = 1: pos = 11
// - in[2] = 1 (and in[3] = 0): pos = 10
// - in[1] = 1 (and in[3:2] = 00): pos = 01
// - in[0] = 1 (and in[3:1] = 000): pos = 00
// - in = 0000: pos = 00