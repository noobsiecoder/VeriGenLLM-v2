// This is a problem dealing with vectors in Verilog
module vector_gates ( 
    input [2:0] a,
    input [2:0] b,
    output [2:0] out_or_bitwise,
    output out_or_logical,
    output [5:0] out_not );
// Build a circuit that has 3 outputs:
// 1) OR Gate
// 2) Logical Gate
// 3) NOT Gate
// Note: Use only `a` and `b` as inputs for each outputs
// Hint: For output NOT gate, split bits between each input, negate them, and store them seperately, Use input `a` at output[2:0] split and `b` for the last memory
// Insert code here