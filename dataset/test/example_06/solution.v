module half_adder (
  input wire a,
  input wire b,
  output wire sum,
  output wire carry
);

  assign sum = a ^ b;       // XOR for sum
  assign carry = a & b;     // AND for carry

endmodule
