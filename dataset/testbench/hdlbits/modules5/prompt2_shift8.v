// Build a 3-stage 8-bit shift register with multiplexed output
module top_module_shift (
    input clk,
    input [7:0] d,
    input [1:0] sel,
    output reg [7:0] q
);
// Instantiate three my_dff8 modules in series
// Note: my_dff8(input clk, input [7:0] d, output reg [7:0] q)
// Create intermediate wires between stages
// Use a multiplexer to select output based on sel:
//   sel=0: output = input d
//   sel=1: output = first stage
//   sel=2: output = second stage  
//   sel=3: output = third stage
// Insert code here