module abro_fsm (
  input wire clk,
  input wire rst,
  input wire A,
  input wire B,
  input wire R,
  output reg O
);

  // State encoding
  parameter IDLE   = 2'b00;
  parameter A_SEEN = 2'b01;
  parameter B_SEEN = 2'b10;
  parameter DONE   = 2'b1;

  reg[1:0] current_state, next_state;

  // Sequential state register
  always @(posedge clk or posedge rst) begin
    if (rst || R)
      current_state <= IDLE;
    else
      current_state <= next_state;
  end

  // Next state logic
  always @(*) begin
    case (current_state)
      IDLE: begin
        if (A && B)
          next_state = DONE;
        else if (A)
          next_state = A_SEEN;
        else if (B)
          next_state = B_SEEN;
        else
          next_state = IDLE;
      end

      A_SEEN: next_state = B ? DONE : A_SEEN;
      B_SEEN: next_state = A ? DONE : B_SEEN;
      DONE:   next_state = DONE;

      default: next_state = IDLE;
    endcase
  end

  // Output logic (Moore machine: output depends only on state)
  always @(*) begin
    O = (current_state == DONE);
  end

endmodule
