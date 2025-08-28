module xnor_matrix (
    input a, b, c, d, e,
    output [24:0] out
);

    assign out = ~{ {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} }
        ^ { {5{a,b,c,d,e}} };

    // There are other ways to solve this question.
    // 1) assign each bit (totally 24 lines!)
    // 2) 2-D loop
    // 3) Procedural style (always)

endmodule