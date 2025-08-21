module top_module ( output one );

    assign one = 1'b1; // value 1 assigned to variable 'one'
    // another acceptable value is 1'd1
    // NOTE: plain 1 is discouraged as it induces noise to the training data
    // and represents a 32-bit value implicitly truncated to 1-bit

endmodule