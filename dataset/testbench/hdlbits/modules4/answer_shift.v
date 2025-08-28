module top_module_shift ( input clk, input d, output q );

    wire sig_a, sig_b;

    my_dff flop1(
        .clk(clk),
        .d(d),
        .q(sig_a)
    );

    my_dff flop2(
        .clk(clk),
        .d(sig_a),
        .q(sig_b)
    );

    my_dff flop3(
        .clk(clk),
        .d(sig_b),
        .q(q)
    );

endmodule