module up_counter (
  input wire clk,
  input wire rst,
  input wire en,
  output reg [3:0] count  // 4-bit counter: counts from 0 to 15
);

  always @(posedge clk or posedge rst) begin
    if (rst)
      count <= 4'd0;           // Reset to 0
    else if (en)
      count <= count + 1;      // Increment only if enable is high
  end

endmodule
