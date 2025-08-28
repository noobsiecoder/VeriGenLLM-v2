// This is a 3-stage (D flip-flop) shift register module  
module top_module_shift ( 
    input clk, 
    input d, 
    output q );
// Import three my_dff modules to create a shift register chain
// Note: my_dff(input clk, input d, output q)  
// Requirements:
// 1. Use instance names: flop1, flop2, flop3
// 2. Create intermediate wires: sig_a (between flop1 and flop2), sig_b (between flop2 and flop3)
// 3. Connect: d -> flop1 -> sig_a -> flop2 -> sig_b -> flop3 -> q
// 4. All flops share the same clock signal
// 5. Use named port connections
// Insert code here