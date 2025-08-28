// Build a 32-bit adder using two cascaded 16-bit adders
module top_module_add (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
// Requirements:
// 1. Declare wires: cin (set to 0), cout1 (carry between adders), cout2 (unused)
// 2. Instantiate first add16 as 'adder1':
//    - Inputs: a[15:0], b[15:0], cin (=0)
//    - Outputs: sum[15:0], cout1
// 3. Instantiate second add16 as 'adder2':
//    - Inputs: a[31:16], b[31:16], cout1 (from first adder)
//    - Outputs: sum[31:16], cout2
// 4. Connect the carry-out of adder1 to carry-in of adder2
// Note: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )
// The module performs: 32-bit a + b (no carry-in or carry-out)
// Insert code here