// Create a hierarchical 32-bit adder with two levels
module top_module_fadd (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
// 1. Instantiate two add16 modules for 32-bit addition
// 2. Connect carry from first to second add16
// Note: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )

endmodule

// Create a 1-bit full adder module
module add1 ( input a, input b, input cin, output sum, output cout );
// Implement full adder logic: sum = a XOR b XOR cin
// cout = majority function
// Insert code here