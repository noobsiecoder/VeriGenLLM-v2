// More verilog features
module top_module_bcadd100 (
    input [399:0] a, b,
    input cin,
    output cout,
    output [399:0] sum
);
// Instantiate 100 copies of bcd_fadd to create a 100-digit BCD ripple-carry adder. Your adder should add two 100-digit BCD numbers (packed into 400-bit vectors) and a carry-in to produce a 100-digit sum and carry out.
//
// Hint: `bcd_fadd` already provided in this question
//
// module bcd_fadd (
//     input [3:0] a,
//     input [3:0] b,
//     input     cin,
//     output   cout,
//     output [3:0] sum );
//
// Requirements:
// 1. Create ripple-carry adder for 100 BCD digits
// 2. Each BCD digit uses 4 bits (values 0-9)
// 3. Total: 100 digits × 4 bits = 400 bits
//
// Bit mapping for BCD digits:
// - Digit 0: a[3:0], b[3:0] → sum[3:0]
// - Digit 1: a[7:4], b[7:4] → sum[7:4]
// - Digit 2: a[11:8], b[11:8] → sum[11:8]
// - ...
// - Digit 99: a[399:396], b[399:396] → sum[399:396]
//
// Note: BCD addition differs from binary:
// - Valid BCD digits: 0000 to 1001 (0-9)
// - If sum > 9, add 6 to correct (handled by bcd_fadd)
// Insert code here