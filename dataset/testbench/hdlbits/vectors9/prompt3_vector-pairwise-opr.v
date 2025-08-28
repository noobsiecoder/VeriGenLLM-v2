// This is a problem dealing with vectors in Verilog
module vector_pairwise_operation (
    input a, b, c, d, e,
    output [24:0] out );
// Given five 1-bit signals (a, b, c, d, and e), compute all 25 pairwise one-bit comparisons in the 25-bit output vector; The output should be 1 if the two bits being compared are equal
// Note:
// out[24] = ~a ^ a;   // a == a, so out[24] is always 1.
// out[23] = ~a ^ b;
// out[22] = ~a ^ c;
// ...
// out[ 1] = ~e ^ d;
// out[ 0] = ~e ^ e;
// Insert code here
