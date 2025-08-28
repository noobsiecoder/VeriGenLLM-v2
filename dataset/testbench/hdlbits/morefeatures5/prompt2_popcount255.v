// More verilog features
module top_module_popcount255 ( 
    input [254:0] in,
    output reg [7:on] out
);
// A "population count" circuit counts the number of '1's in an input vector; Build a population count circuit for a 255-bit input vector
// Count the number of 1's in the input vector
// Use procedural assignment in always block
// Insert code here