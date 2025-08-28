// Build 100-input logic gates using reduction operators
module top_module_gates100 (
    input [99:0] in,
    output out_and,
    output out_or,
    output out_xor
);
// Requirements:
// 1. Create three different 100-input gates
// 2. Use reduction operators for efficient implementation
//
// Implementation:
// assign out_and = &in;  // AND reduction
// assign out_or = |in;   // OR reduction
// assign out_xor = ^in;  // XOR reduction
//
// Reduction operator behavior:
// - &in = in[99] & in[98] & ... & in[1] & in[0]
//   Result: 1 only if ALL bits are 1
//
// - |in = in[99] | in[98] | ... | in[1] | in[0]
//   Result: 1 if ANY bit is 1
//
// - ^in = in[99] ^ in[98] ^ ... ^ in[1] ^ in[0]
//   Result: 1 if ODD number of bits are 1 (parity)
//
// Examples:
// - in = 100'h0 → AND=0, OR=0, XOR=0
// - in = {100{1'b1}} → AND=1, OR=1, XOR=0 (100 is even)
// - in = 100'h1 → AND=0, OR=1, XOR=1
// Insert code here