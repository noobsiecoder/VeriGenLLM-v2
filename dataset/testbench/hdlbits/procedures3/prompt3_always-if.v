// Build a 2-to-1 mux with AND-gated select condition
module top_module_always (
    input a,
    input b,
    input sel_b1,
    input sel_b2,
    output wire out_assign,
    output reg out_always
);
// Requirements:
// 1. Select logic: Choose b when (sel_b1 == 1 AND sel_b2 == 1)
//                  Otherwise choose a
//
// 2. Continuous assignment implementation:
//    assign out_assign = (sel_b1 && sel_b2) ? b : a;
//    - Uses ternary operator for concise expression
//    - && performs logical AND
//
// 3. Procedural implementation:
//    always @(*) begin
//        if (sel_b1 && sel_b2) begin
//            out_always = b;
//        end
//        else begin
//            out_always = a;
//        end
//    end
//    - Uses if-else structure
//    - Same logic as assign method
//
// Note: Both create identical combinational logic
// The AND condition ensures b is selected only when BOTH selects are true
// Insert code here