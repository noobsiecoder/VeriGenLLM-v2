// Build a 2-to-1 multiplexer with two select signals
module top_module_always (
    input a,
    input b,
    input sel_b1,
    input sel_b2,
    output wire out_assign,
    output reg out_always
);
// Choose b if BOTH sel_b1 AND sel_b2 are true
// Otherwise choose a
// Implement twice: using assign and using if-else
// Insert code here