module priority_encoder_3bit (
  input  [2:0] in,
  output reg [1:0] out,
  output reg valid
);

  always @(*) begin
    casez (in)
      3'b1??: begin out = 2'b10; valid = 1; end  // Highest priority: in[2]
      3'b01?: begin out = 2'b01; valid = 1; end  // Middle priority: in[1]
      3'b001: begin out = 2'b00; valid = 1; end  // Lowest priority: in[0]
      default: begin out = 2'b00; valid = 0; end // No input active
    endcase
  end

endmodule
