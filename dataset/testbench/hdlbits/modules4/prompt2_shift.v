// This is a 3-stage shift register module
module top_module_shift ( 
    input clk, 
    input d, 
    output q );
// Import three my_dff modules connected in series
// Note: my_dff(input clk, input d, output q)
// Use instance names: flop1, flop2, flop3
// Insert code here