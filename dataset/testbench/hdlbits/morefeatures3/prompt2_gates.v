// Build 100-input combinational gates
module top_module_gates100 (
    input [99:0] in,
    output out_and,
    output out_or,
    output out_xor
);
// Implement three gates:
// - out_and: 100-input AND gate (all bits must be 1)
// - out_or: 100-input OR gate (at least one bit must be 1)
// - out_xor: 100-input XOR gate (odd number of 1s)
// Use Verilog reduction operators: &, |, ^
// Insert code here