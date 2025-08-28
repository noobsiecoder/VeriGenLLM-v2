// Build a PS/2 keyboard scancode decoder for arrow keys
module top_module_always_nolatches (
    input [15:0] scancode,
    output reg left,
    output reg down,
    output reg right,
    output reg up
);
// Requirements:
// 1. Decode PS/2 keyboard scancodes for arrow keys
// 2. Scancode mapping:
//    - 16'he06b → left arrow pressed (left = 1)
//    - 16'he072 → down arrow pressed (down = 1)
//    - 16'he074 → right arrow pressed (right = 1)
//    - 16'he075 → up arrow pressed (up = 1)
//    - Any other scancode → no key pressed (all outputs = 0)
//
// 3. Only one output should be high at a time
// 4. Prevent latches by initializing all outputs to 0
//
// Implementation:
// always @(*) begin
//     // Initialize all outputs to 0 (prevents latches)
//     up = 1'b0;
//     down = 1'b0;
//     right = 1'b0;
//     left = 1'b0;
//     
//     case(scancode)
//         16'he06b: left = 1'b1;
//         16'he072: down = 1'b1;
//         16'he074: right = 1'b1;
//         16'he075: up = 1'b1;
//         default: ;  // All remain 0
//     endcase
// end
//
// Note: Initializing outputs before case prevents latches