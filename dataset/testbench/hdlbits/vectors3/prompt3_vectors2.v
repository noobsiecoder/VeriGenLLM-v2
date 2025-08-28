// This is a problem dealing with vectors in Verilog
module top_module( 
    input wire [15:0] in,
    output wire [7:0] out_hi,
    output wire [7:0] out_lo
);
// Build a circuit that will reverse the byte ordering of the 4-byte word (endianness conversion)
// Hint:
// As the question says, per 4-bit of the input can be reversed.
// This means, for `out[7:0] := in[31:24]` and so on...
// Insert code here