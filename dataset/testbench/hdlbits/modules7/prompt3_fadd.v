// Create a two-level hierarchical 32-bit adder
module top_module_fadd (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
// Requirements:
// 1. Declare wires: cout1 (carry between modules), cout2 (unused)
// 2. Instantiate first add16 as 'adder1':
//    - Inputs: a[15:0], b[15:0], cin=0
//    - Outputs: sum[15:0], cout1
// 3. Instantiate second add16 as 'adder2':
//    - Inputs: a[31:16], b[31:16], cin=cout1
//    - Outputs: sum[31:16], cout2
// Note: add16 internally uses 16 add1 modules
// Module declaration: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )

endmodule

// Create a 1-bit full adder module
module add1 ( input a, input b, input cin, output sum, output cout );
// Requirements:
// 1. Implement sum output using XOR gates:
//    sum = a XOR b XOR cin
// 2. Implement carry output using majority logic:
//    cout = (a AND b) OR (a AND cin) OR (b AND cin)
// 3. Use continuous assignments (assign statements)
// This creates the basic building block for the ripple-carry adder
// Insert code here