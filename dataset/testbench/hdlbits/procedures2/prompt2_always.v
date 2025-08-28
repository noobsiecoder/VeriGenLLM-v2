// Build an XOR gate three ways: immediate and delayed
module top_module_always (
    input clk,
    input a,
    input b,
    output wire out_assign,
    output reg out_always_comb,
    output reg out_always_ff
);
// 1. Use assign statement (combinational)
// 2. Use always @(*) block (combinational)
// 3. Use always @(posedge clk) block (sequential with flip-flop)
// All implement XOR function: a ^ b
// Insert code here