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
// Use case statement to select data inputs
// Output 0 for invalid selections