// This is a problem dealing with vectors in Verilog
module top_module ( 
    input wire [2:0] vec,
    output wire [2:0] outv,
    output wire o2,
    output wire o1,
    output wire o0 );
// From the input 3-bit wire(s), transit the whole signal to the corresponding output, and each bit to each 1-bit output
// HINT:
// 1) vec -> outv
// 2) each of vec bit assigned to other 1-bit output
// Insert code here