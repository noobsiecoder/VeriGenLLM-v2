// Build a shift register with selectable output
module top_module_shift (
    input clk,
    input [7:0] d,
    input reg [1:0] sel
// Create a 3-stage 8-bit shift register using my_dff8
// Add output selection logic
// Note: my_dff8(input clk, input [7:0] d, output reg [7:0] q)
// Insert code here