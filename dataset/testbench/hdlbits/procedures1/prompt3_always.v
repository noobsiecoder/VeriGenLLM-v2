// Build an AND gate using both implementation methods
module top_module_always (
    input a,
    input b,
    output wire out_assign,
    output reg out_alwaysblock
);
// Requirements:
// 1. Continuous assignment method:
//    - Use: assign out_assign = a & b;
//    - This creates a wire that continuously reflects a AND b
//    
// 2. Procedural assignment method:
//    - Use always block with sensitivity list: always @(*)
//    - Inside block: out_alwaysblock = a & b;
//    - The @(*) means sensitive to all signals on RHS
//    - Output must be declared as 'reg' type
//
// Note: Both methods create identical combinational logic
// The assign is simpler for basic logic
// The always block is more flexible for complex logic
// Insert code here