// This is a problem dealing with vectors in Verilog
module vector_concat (
    input [4:0] a, b, c, d, e, f,
    output [7:0] w, x, y, z );
// Given several input vectors, concatenate them together then split them up into several output vectors; The output should be a concatenation of the input vectors followed by two 1 bits
// There are six 5-bit input vectors: a, b, c, d, e, and f, for a total of 30 bits of input. 
// Note: There are four 8-bit output vectors: w, x, y, and z, for 32 bits of output.
// Insert code here
