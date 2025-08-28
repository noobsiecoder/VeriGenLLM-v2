// Build a 3-stage 8-bit shift register with multiplexed output
module top_module_shift (
    input clk,
    input [7:0] d,
    input [1:0] sel,
    output reg [7:0] q
);
// Requirements:
// 1. Declare three 8-bit wires: sig_a, sig_b, sig_c
// 2. Instantiate three my_dff8 modules:
//    - flop1: d input -> sig_a output
//    - flop2: sig_a input -> sig_b output  
//    - flop3: sig_b input -> sig_c output
// 3. All flops use the same clock signal
// 4. Create combinational output multiplexer using always block:
//    - sel=00: q = d (bypass all delays)
//    - sel=01: q = sig_a (1 clock delay)
//    - sel=10: q = sig_b (2 clock delays)
//    - sel=11: q = sig_c (3 clock delays)
// 5. Use case statement for the multiplexer
// Note: my_dff8(input clk, input [7:0] d, output reg [7:0] q)
// Insert code here