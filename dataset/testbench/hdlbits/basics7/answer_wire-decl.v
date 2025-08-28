`default_nettype none
module wire_decl_gate (
    input a,
    input b,
    input c,
    input d,
    output out,
    output out_n
);

    wire tmp_and_top, tmp_and_btm, tmp_or_mdl; // assign wires

    assign tmp_and_top = a & b; // Top wire with AND GATE
    assign tmp_and_btm = c & d; // Bottom wire with AND GATE

    assign tmp_or_mdl = tmp_and_top | tmp_and_btm; // Middle wire with OR GATE
    
    assign out = tmp_or_mdl; // Output 1
    assign out_n = ~(tmp_or_mdl); // Output 2 - negated

endmodule