// This is a problem dealing with vectors in Verilog
`default_nettype none // This line disable implicit nets by cautioning the compiler
module top_module( 
    input wire [15:0] in,
    output wire [7:0] out_hi,
    output wire [7:0] out_lo
);
// Build a combinational circuit that splits an input half-word
// Insert code here
