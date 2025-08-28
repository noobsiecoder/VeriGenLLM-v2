// Build a PS/2 scancode decoder for game arrow keys
module top_module_always_nolatches (
    input [15:0] scancode,
    output reg left,
    output reg down,
    output reg right,
    output reg up
);
// Map scancodes to arrow keys:
// - 16'he06b: left arrow (left = 1, others = 0)
// - 16'he072: down arrow (down = 1, others = 0)
// - 16'he074: right arrow (right = 1, others = 0)
// - 16'he075: up arrow (up = 1, others = 0)
// - Any other value: all outputs = 0
// Use case statement and initialize all outputs to prevent latches