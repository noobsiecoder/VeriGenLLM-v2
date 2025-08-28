// This is a wire declaration problem
`default_nettype none
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
//
// NOTE: `default_nettype none is used.
// Insert code here
