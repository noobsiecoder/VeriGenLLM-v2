module two_state_fsm (
  input wire clk,
  input wire rst,
  input wire in,
  output reg out
);

  // State encoding
  parameter STATE_0 = 1'b0;
  parameter STATE_1 = 1'b1;
  
  reg current_state, next_state;

  // State register
  always @(posedge clk or posedge rst) begin
    if (rst)
      current_state <= STATE_0;
    else
      current_state <= next_state;
  end

  // Next state logic
  always @(*) begin
    case (current_state)
      STATE_0: next_state = in ? STATE_1 : STATE_0;
      STATE_1: next_state = in ? STATE_0 : STATE_1;
      default: next_state = STATE_0;
    endcase
  end

  // Output logic
  always @(*) begin
    case (current_state)
      STATE_0: out = 1'b0;
      STATE_1: out = 1'b1;
      default: out = 1'b0;
    endcase
  end

endmodule