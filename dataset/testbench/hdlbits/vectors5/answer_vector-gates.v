module four_bit_vector_gates ( 
    input [3:0] in,
    output out_and,
    output out_or,
    output out_xor
);

    assign out_and = in[0] & in[1] & in[2] & in[3]; // Can also && as it 1-bit
    assign out_or = in[0] | in[1] | in[2] | in[3]; // Can also || as it 1-bit
    assign out_xor = in[0] ^ in[1] ^ in[2] ^ in[3];

endmodule