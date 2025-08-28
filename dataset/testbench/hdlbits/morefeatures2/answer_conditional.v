module top_module_even_parity (
    input [7:0] in,
    output parity
);

    assign parity = ^in;

endmodule