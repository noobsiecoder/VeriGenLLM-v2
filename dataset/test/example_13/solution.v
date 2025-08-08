module and_gate_case (
  input wire a,
  input wire b,
  output reg y
);

  always @(*) begin
    case ({a, b})
      2'b00: y = 1'b0;
      2'b01: y = 1'b0;
      2'b10: y = 1'b0;
      2'b11: y = 1'b1;
    endcase
  end

endmodule
