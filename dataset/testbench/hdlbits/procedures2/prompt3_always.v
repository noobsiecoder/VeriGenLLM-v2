// Build an XOR gate using three implementation methods
module top_module_always (
    input clk,
    input a,
    input b,
    output wire out_assign,
    output reg out_always_comb,
    output reg out_always_ff
);
// Requirements:
// 1. Continuous assignment (combinational):
//    assign out_assign = a ^ b;
//    - Creates pure combinational logic
//    - Output changes immediately with inputs
//    
// 2. Combinational always block:
//    always @(*) begin
//        out_always_comb = a ^ b;
//    end
//    - Also creates pure combinational logic
//    - Functionally identical to assign method
//    - The @(*) triggers on any input change
//    
// 3. Sequential always block:
//    always @(posedge clk) begin
//        out_always_ff = a ^ b;
//    end
//    - Creates a D flip-flop followed by XOR logic
//    - Output updates only on positive clock edge
//    - Delayed by one clock cycle
//    - Useful for synchronizing signals
//
// Note: First two create identical circuits, third adds memory
// Insert code here