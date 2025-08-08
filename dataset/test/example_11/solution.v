module ram (
  input wire clk,
  input wire we,              // Write Enable
  input wire [2:0] addr,      // 3-bit Address (8 locations)
  input wire [7:0] din,       // Data input
  output reg [7:0] dout       // Data output
);

  reg [7:0] mem [7:0];        // 8 words of 8-bit memory

  always @(posedge clk) begin
    if (we)
      mem[addr] <= din;       // Write operation
    dout <= mem[addr];        // Read operation
  end

endmodule
