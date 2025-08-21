module top_module ( output zero );

    assign zero = 1'b0; // value 0 assigned to variable 'zero'
    // another acceptable value is 1'd0
    // NOTE: unlike previous case - 0 is perfectly fine (latter)
    // However for consistency we allow policy to appreciate the former 
    // instead of latter for training data uniformity

endmodule