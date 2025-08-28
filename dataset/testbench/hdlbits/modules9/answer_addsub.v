module top_module_addsub (
    input [31:0] a,
    input [31:0] b,
    input sub,
    output [31:0] sum
);

    wire cout1, cout2;
    wire [31:0] xor_sig;

    assign xor_sig = {32{sub}}^b;
    add16 adder1(
        a[15:0], xor_sig[15:0], sub, sum[15:0], cout1
    );

    add16 adder2(
        a[31:16], xor_sig[31:16], cout1, sum[31:16], cout2
    );

endmodule