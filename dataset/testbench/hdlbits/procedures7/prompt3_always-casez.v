// Build a 4-bit priority encoder with casez
module top_module_always_case (
    input [3:0] in,
    output reg [1:0] pos
);
// Requirements:
// 1. Priority encoder finds position of highest set bit
// 2. Bit 3 has highest priority, bit 0 has lowest
// 3. Output encoding:
//    - pos = 11 when in[3] = 1 (regardless of other bits)
//    - pos = 10 when in[3] = 0 and in[2] = 1
//    - pos = 01 when in[3:2] = 00 and in[1] = 1
//    - pos = 00 when in[3:1] = 000 and in[0] = 1
//    - pos = 00 when in = 0000 (no bits set)
//
// Implementation:
// always @(*) begin
//     casez(in)
//         4'b1???: pos = 2'b11;  // Bit 3 set, others don't care
//         4'b01??: pos = 2'b10;  // Bit 3=0, bit 2=1, others don't care
//         4'b001?: pos = 2'b01;  // Bits 3,2=0, bit 1=1, bit 0 don't care
//         4'b0001: pos = 2'b00;  // Only bit 0 set
//         default: pos = 2'b00;  // No bits set (0000)
//     endcase
// end
//
// Note: casez treats ? as don't care in case patterns