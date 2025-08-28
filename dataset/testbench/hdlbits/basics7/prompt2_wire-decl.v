// This is a xnor gate module
module wire_decl_gate (
    input a,
    input b,
    input c,
    input d,
    output out,
    output out_n
);
// First four signals (each two), go through two AND gates respectively.
// a, b -> top AND gate     = X
// c, d -> bottom AND gate  = Y
// Then the two signals coming out, goes through a OR gate.
// X, Y -> middle OR gate   = Z
//
// NOTE: `default_nettype none is used. This is a compiler directive line and it signals the compiler to not automatically create nets. Every signal must be explicitly declared.
// Insert code here