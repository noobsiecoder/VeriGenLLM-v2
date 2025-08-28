module top_module_adder100i (
    input [99:0] a, b,
    input cin,
    output [99:0] cout,
    output [99:0] sum
);

    genvar i;
    generate
        for (i = 0; i < 100; i = i + 1) begin : adder100i
            if (i == 1'b0)
                add1 adder(a[0], b[0], cin, cout[0], sum[0]);
            else
                add1 adder(a[i], b[i], cout[i - 1], cout[i], sum[i]);
        end
    endgenerate

endmodule

module add1(
    input a,
    input b,
    input cin,
    output cout,
    output sum
);

    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (a & cin) | (b & cin);

endmodule