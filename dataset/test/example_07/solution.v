module counter_1_to_12 (
  input wire clk,
  input wire rst,       // Active-high synchronous reset
  output reg [3:0] count
);

  always @(posedge clk or posedge rst) begin
    if (rst)
      count <= 4'd1;
    else if (count == 4'd12)
      count <= 4'd1;
    else
      count <= count + 4'd1;
  end

endmodule
