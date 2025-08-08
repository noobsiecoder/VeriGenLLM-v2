module sequence_detector_101 (
  input wire clk,
  input wire rst,
  input wire in,
  output reg detected
);
  // State encoding
  parameter S0 = 2'b00;
  parameter S1 = 2'b01;
  parameter S2 = 2'b10;
  
  reg [1:0] current_state, next_state;
  
  // State register
  always @(posedge clk or posedge rst) begin
    if (rst)
      current_state <= S0;
    else
      current_state <= next_state;
  end
  
  // Next state logic
  always @(*) begin
    case (current_state)
      S0: next_state = (in == 1) ? S1 : S0;
      S1: next_state = (in == 0) ? S2 : S1;
      S2: next_state = (in == 1) ? S1 : S0;
      default: next_state = S0;
    endcase
  end
  
  // Output logic
  always @(*) begin
    detected = (current_state == S2 && in == 1);
  end
endmodule