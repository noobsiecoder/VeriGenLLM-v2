module top_module_cseladd (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);

    wire cout, cout2_top, cout2_btm;
    wire [15:0] sum1, sum2;

    add16 adder(
        a[15:0], b[15:0], 1'b0, sum[15:0], cout
    );

    add16 adder2_top(
        a[31:16], b[31:16], 1'b0, sum1, cout2_top
    );

    add16 adder2_btm(
        a[31:16], b[31:16], 1'b1, sum2, cout2_btm
    );

    assign sum[31:16] = (cout == 0) ? sum1 : sum2; 
    // Note: always block cannot be used for sum as it is a wire in continuous form.
    // To use always, sum must be initialised as `output reg [31:0] sum`

endmodule