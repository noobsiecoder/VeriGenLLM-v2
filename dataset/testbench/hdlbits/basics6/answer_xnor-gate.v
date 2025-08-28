module xnor_gate ( input a, input b, output out );

    assign out =  ~(a ^ b); // assign value inverted to output

endmodule