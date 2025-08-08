module signed_adder_8bit (
  input wire [7:0] a,
  input wire [7:0] b,
  output wire [7:0] sum,
  output wire overflow
);

  assign sum = a + b;

  // Overflow occurs if:
  // - a and b have the same sign
  // - but sum has the opposite sign
  assign overflow = (~a[7] & ~b[7] &  sum[7]) |  // Positive + Positive = Negative (overflow)
                    ( a[7] &  b[7] & ~sum[7]);   // Negative + Negative = Positive (overflow)

endmodule
