module vector_concat (
    input [4:0] a, b, c, d, e, f,
    output [7:0] w, x, y, z
);

    wire [31:0] concat_out;
    assign concat_out = {
        a[4:0], 
        b[4:0],
        c[4:0],
        d[4:0],
        e[4:0],
        f[4:0],
        2'b11 // two 1-bits
    };
    // assign outputs
    assign w = concat_out[31:24];
    assign x = concat_out[23:16];
    assign y = concat_out[15:8];
    assign z = concat_out[7:0];  

endmodule