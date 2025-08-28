`default_nettype none // This line disable implicit nets by cautioning the compiler
module top_module(
    input wire [15:0] in,
    output wire [7:0] out_hi,
    output wire [7:0] out_lo
);

    assign out_hi = in[15:8];
    assign out_lo = in[7:0];

endmodule