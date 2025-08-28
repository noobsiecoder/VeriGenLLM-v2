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
// At the end:
// 1) Output out, comes out directly after the OR gate
// 2) Output out_n, comes out after negated the Z signal
//
// NOTE: `default_nettype none is used. This is a compiler directive line and it signals the compiler to not automatically create nets. Every signal must be explicitly declared.
// For instance:
// - In Verilog, if you use a signal name without declaring it, the compiler by default assumes it’s a wire.
// ```verilog
// ...
// assign y = a & b;  // If 'y' wasn't declared, it's implicitly a wire.
// ```
// This "auto-wire creation" can hide bugs (like typos in signal names).
// Hence, it is a good caution to include it in code.
// Insert code here