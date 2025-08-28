// Build a PS/2 keyboard scancode decoder for arrow keys
module top_module_always_nolatches (
    input [15:0] scancode,
    output reg left,
    output reg down,
    output reg right,
    output reg up
);
// Decode these scancodes:
// 16'he06b = left, 16'he072 = down
// 16'he074 = right, 16'he075 = up
// Avoid creating latches