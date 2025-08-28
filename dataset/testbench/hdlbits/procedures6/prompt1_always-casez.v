// Build a 4-bit priority encoder
module top_module_always_case (
    input [3:0] in,
    output reg [1:0] pos
);
// Use casez statement
// Output position of highest priority bit
// Output 00 if no bits are set