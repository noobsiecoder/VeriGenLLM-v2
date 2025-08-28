// Build a 6-to-1 multiplexer using case statement
module top_module_always_case (
    input [2:0] sel, 
    input [3:0] data0,
    input [3:0] data1,
    input [3:0] data2,
    input [3:0] data3,
    input [3:0] data4,
    input [3:0] data5,
    output reg [3:0] out
);
// Requirements:
// 1. Use always @(*) block with case statement
// 2. Map selections:
//    - sel = 3'b000 (0): out = data0
//    - sel = 3'b001 (1): out = data1
//    - sel = 3'b010 (2): out = data2
//    - sel = 3'b011 (3): out = data3
//    - sel = 3'b100 (4): out = data4
//    - sel = 3'b101 (5): out = data5
//    - sel = 3'b110 (6): out = 4'b0000
//    - sel = 3'b111 (7): out = 4'b0000
// 3. Use default case for values 6 and 7
// 4. Note: All data paths are 4 bits wide
//
// Syntax:
// always @(*) begin
//     case(sel)
//         3'b000: out = data0;
//         // ... more cases
//         default: out = 4'b0;
//     endcase
// end