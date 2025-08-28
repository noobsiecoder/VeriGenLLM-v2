module top_module_shift (
    input clk,
    input [7:0] d,
    input [1:0] sel,
    output reg [7:0] q
);

    wire [7:0] sig_a, sig_b, sig_c;

    my_dff8 flop1(
        .clk(clk),
        .d(d),
        .q(sig_a)
    );

    my_dff8 flop2(
        .clk(clk),
        .d(sig_a),
        .q(sig_b)
    );

    my_dff8 flop3(
        .clk(clk),
        .d(sig_b),
        .q(sig_c)
    );

    always @ (*) begin
        case(sel)
            0: q = d;
            1: q = sig_a;
            2: q = sig_b;
            3: q = sig_c;
        endcase
    end

endmodule