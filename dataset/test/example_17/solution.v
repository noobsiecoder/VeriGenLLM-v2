module arithmetic_shift_register (
  input wire clk,
  input wire rst,
  input wire shift_left,     // 1: shift left; 0: shift right
  input wire enable,         // Enables shift operation
  input wire [63:0] data_in, // Load value
  input wire load,           // Load data when high
  output reg [63:0] data_out
);

  always @(posedge clk or posedge rst) begin
    if (rst)
      data_out <= 64'd0;
    else if (load)
      data_out <= data_in;
    else if (enable) begin
      if (shift_left)
        data_out <= data_out <<< 1;  // Arithmetic shift left
      else
        data_out <= data_out >>> 1;  // Arithmetic shift right (preserves sign bit)
    end
  end

endmodule
