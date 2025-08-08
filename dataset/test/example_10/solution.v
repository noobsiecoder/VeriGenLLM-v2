module shift_and_rotate (
  input wire [7:0] in,
  input wire sel,        // 0: shift left, 1: rotate left
  output reg [7:0] out
);

  always @(*) begin
    if (sel == 1'b0)
      out = in << 1;  // Logical shift left by 1
    else
      out = {in[6:0], in[7]};  // Rotate left by 1
  end

endmodule
