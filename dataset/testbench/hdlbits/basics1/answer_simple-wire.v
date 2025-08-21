module simple_wire ( input in, output out );

    assign out = in; // Simple wire connection from input to output
    // NOTE: This creates a continuous assignment (combinational logic)

endmodule