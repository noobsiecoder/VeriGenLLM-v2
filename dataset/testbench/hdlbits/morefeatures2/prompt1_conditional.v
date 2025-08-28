// Create an even parity bit generator
module top_module_even_parity (
    input [7:0] in,
    output parity
);
// Generate parity bit for error detection
// Use XOR of all bits