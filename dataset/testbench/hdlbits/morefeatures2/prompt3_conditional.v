// Build an even parity bit generator for error detection
module top_module_even_parity (
    input [7:0] in,
    output parity
);
// Requirements:
// 1. Generate a parity bit for 8-bit data
// 2. Use even parity scheme:
//    - Count number of 1s in the data
//    - If odd number of 1s: parity = 1
//    - If even number of 1s: parity = 0
//    - Result: data + parity has even number of 1s
//
// 3. Implementation: parity = XOR of all data bits
//    assign parity = ^in;
//    
// The ^ operator is the XOR reduction operator:
// ^in = in[7] ^ in[6] ^ in[5] ^ in[4] ^ in[3] ^ in[2] ^ in[1] ^ in[0]
//
// Examples:
// - in = 8'b00000000 → 0 ones → parity = 0
// - in = 8'b00000001 → 1 one → parity = 1
// - in = 8'b00000011 → 2 ones → parity = 0
// - in = 8'b11111111 → 8 ones → parity = 0
//
// Error detection: If received data + parity has odd number
// of 1s, a single-bit error occurred during transmission