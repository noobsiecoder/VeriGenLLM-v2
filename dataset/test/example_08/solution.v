module lfsr_random (
  input wire clk,
  input wire rst,
  output reg [5:0] lfsr_out
);

  wire feedback;

  // XOR taps at bit positions 3 and 5 => bits [2] and [4]
  assign feedback = lfsr_out[2] ^ lfsr_out[4];

  always @(posedge clk or posedge rst) begin
    if (rst)
      lfsr_out <= 6'b000001;  // Initial seed must not be all zeros
    else
      lfsr_out <= {lfsr_out[4:0], feedback};  // Shift left and insert feedback
  end

endmodule
