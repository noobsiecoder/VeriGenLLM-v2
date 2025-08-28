// Build a 32-bit adder-subtractor using two's complement
module top_module_addsub (
    input [31:0] a,
    input [31:0] b,
    input sub,
    output [31:0] sum
);
// Requirements:
// 1. Declare wires: cout1, cout2, and xor_sig[31:0]
// 2. Create XOR operation:
//    - When sub=0: xor_sig = b (pass through for addition)
//    - When sub=1: xor_sig = ~b (invert for subtraction)
//    - Use: assign xor_sig = {32{sub}} ^ b;
// 3. Instantiate first add16 as 'adder1':
//    - Inputs: a[15:0], xor_sig[15:0], cin=sub
//    - Outputs: sum[15:0], cout1
// 4. Instantiate second add16 as 'adder2':
//    - Inputs: a[31:16], xor_sig[31:16], cin=cout1
//    - Outputs: sum[31:16], cout2
// Note: Subtraction works because a-b = a+(-b) = a+(~b+1)
// The 'sub' signal provides the +1 through carry-in
// Module: add16 ( input[15:0] a, input[15:0] b, input cin, output[15:0] sum, output cout )
// Insert code here