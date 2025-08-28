// Build an even parity generator for 8-bit data
module top_module_even_parity (
    input [7:0] in,
    output parity
);
// Even parity: Total number of 1s (data + parity) is even
// Parity bit = XOR of all 8 data bits
// If data has odd number of 1s, parity = 1
// If data has even number of 1s, parity = 0