module and_gate ( input a, input b, output out );

    assign out = a & b; // assign input inverted to output
    // Note: Another method: and(out, a, b);
    // This works since AND, OR, NOT are primitive gate instantiations in Verilog

endmodule