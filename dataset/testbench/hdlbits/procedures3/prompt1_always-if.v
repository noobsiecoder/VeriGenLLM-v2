// Build a 2-to-1 mux with AND condition
module top_module_always (
    input a,
    input b,
    input sel_b1,
    input sel_b2
// Select b when both select signals are true
// Use both assign and always block
// Insert code here