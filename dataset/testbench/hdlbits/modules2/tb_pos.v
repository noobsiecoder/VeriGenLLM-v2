`timescale 1ns/1ps

// First, we need to define mod_a for the testbench to work
// (In real scenario, this would be in a separate file)
// Note: The port order matters here since top_module uses positional connections
module mod_a (
    output wire out1,
    output wire out2,
    input wire in1,
    input wire in2, 
    input wire in3,
    input wire in4
);
    // Example implementation - replace with actual logic
    assign out1 = (in1 & in2) | (in3 & in4);  // Example: OR of two AND operations
    assign out2 = (in1 ^ in2) ^ (in3 ^ in4);  // Example: XOR of all inputs
endmodule

// Testbench for top_module_pos
module tb_top_module_pos();
    // Inputs (driven by testbench)
    reg a, b, c, d;
    
    // Outputs (driven by DUT)
    wire out1, out2;
    
    // Instantiate the Design Under Test (DUT)
    top_module_pos dut (
        .a(a),
        .b(b),
        .c(c),
        .d(d),
        .out1(out1),
        .out2(out2)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected values (will be determined based on mod_a's function)
    reg expected_out1, expected_out2;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing top_module_pos with positional mod_a instantiation");
        $display("Port mapping (positional):");
        $display("  mod_a port 1 (out1) <- top_module.out1");
        $display("  mod_a port 2 (out2) <- top_module.out2");
        $display("  mod_a port 3 (in1)  <- top_module.a");
        $display("  mod_a port 4 (in2)  <- top_module.b");
        $display("  mod_a port 5 (in3)  <- top_module.c");
        $display("  mod_a port 6 (in4)  <- top_module.d");
        $display("===============================================================");
        
        // Test all 16 possible input combinations
        $display("\nTruth Table for all input combinations:");
        $display("---------------------------------------");
        $display("Time | a b c d | out1 out2 | Decimal");
        $display("---------------------------------------");
        
        for (i = 0; i < 16; i = i + 1) begin
            {a, b, c, d} = i[3:0];
            #10;
            $display("%3t  | %b %b %b %b |  %b    %b   |   %2d", 
                $time, a, b, c, d, out1, out2, i);
        end
        
        // Analyze the outputs to determine mod_a's function
        $display("\n===============================================================");
        $display("Function Analysis:");
        $display("---------------------------------------------------------------");
        
        // Test Case 1: All zeros
        {a, b, c, d} = 4'b0000;
        #10;
        $display("All zeros: out1=%b, out2=%b", out1, out2);
        
        // Test Case 2: Single bit high patterns
        $display("\nSingle bit high patterns:");
        {a, b, c, d} = 4'b1000; #10;
        $display("a=1, others=0: out1=%b, out2=%b", out1, out2);
        
        {a, b, c, d} = 4'b0100; #10;
        $display("b=1, others=0: out1=%b, out2=%b", out1, out2);
        
        {a, b, c, d} = 4'b0010; #10;
        $display("c=1, others=0: out1=%b, out2=%b", out1, out2);
        
        {a, b, c, d} = 4'b0001; #10;
        $display("d=1, others=0: out1=%b, out2=%b", out1, out2);
        
        // Test Case 3: Paired inputs
        $display("\nPaired input patterns:");
        {a, b, c, d} = 4'b1100; #10;
        $display("a=1, b=1, c=0, d=0: out1=%b, out2=%b", out1, out2);
        
        {a, b, c, d} = 4'b0011; #10;
        $display("a=0, b=0, c=1, d=1: out1=%b, out2=%b", out1, out2);
        
        {a, b, c, d} = 4'b1010; #10;
        $display("a=1, b=0, c=1, d=0: out1=%b, out2=%b", out1, out2);
        
        {a, b, c, d} = 4'b0101; #10;
        $display("a=0, b=1, c=0, d=1: out1=%b, out2=%b", out1, out2);
        
        // Test Case 4: All ones
        {a, b, c, d} = 4'b1111;
        #10;
        $display("\nAll ones: out1=%b, out2=%b", out1, out2);
        
        // Comprehensive testing with expected values
        $display("\n===============================================================");
        $display("Comprehensive Testing:");
        $display("---------------------------------------------------------------");
        $display("Time | a b c d | out1 out2 | Expected | Result");
        $display("-----------------------------------------------");
        
        for (i = 0; i < 16; i = i + 1) begin
            {a, b, c, d} = i[3:0];
            
            // Calculate expected values based on assumed mod_a logic
            // Adjust these based on actual mod_a implementation
            expected_out1 = (a & b) | (c & d);
            expected_out2 = (a ^ b) ^ (c ^ d);
            
            #10;
            
            if (out1 == expected_out1 && out2 == expected_out2) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
            
            $display("%3t  | %b %b %b %b |  %b    %b   | %b    %b   | %s",
                $time, a, b, c, d, out1, out2, expected_out1, expected_out2,
                (out1 == expected_out1 && out2 == expected_out2) ? "PASS" : "FAIL");
        end
        
        // Signal transition testing
        $display("\n===============================================================");
        $display("Signal Transition Testing:");
        $display("---------------------------------------------------------------");
        
        // Test rapid transitions
        $display("Rapid transitions on all inputs:");
        {a, b, c, d} = 4'b0000;
        #5; {a, b, c, d} = 4'b1111;
        #5; {a, b, c, d} = 4'b0101;
        #5; {a, b, c, d} = 4'b1010;
        #5;
        $display("Final state: a=%b b=%b c=%b d=%b => out1=%b out2=%b", 
            a, b, c, d, out1, out2);
        
        // Gray code sequence (single bit changes)
        $display("\nGray code sequence (single bit changes):");
        {a, b, c, d} = 4'b0000; #10;
        $display("%b%b%b%b => out1=%b out2=%b", a, b, c, d, out1, out2);
        
        {a, b, c, d} = 4'b0001; #10;
        $display("%b%b%b%b => out1=%b out2=%b", a, b, c, d, out1, out2);
        
        {a, b, c, d} = 4'b0011; #10;
        $display("%b%b%b%b => out1=%b out2=%b", a, b, c, d, out1, out2);
        
        {a, b, c, d} = 4'b0010; #10;
        $display("%b%b%b%b => out1=%b out2=%b", a, b, c, d, out1, out2);
        
        // Module connection verification
        $display("\n===============================================================");
        $display("Positional Connection Verification:");
        $display("---------------------------------------------------------------");
        $display("IMPORTANT: This module uses POSITIONAL port connections!");
        $display("The order of ports in mod_a instantiation is critical.");
        $display("\nInstantiation: mod_a instance1(out1, out2, a, b, c, d);");
        $display("This means:");
        $display("  1st port of mod_a connects to out1");
        $display("  2nd port of mod_a connects to out2");
        $display("  3rd port of mod_a connects to a");
        $display("  4th port of mod_a connects to b");
        $display("  5th port of mod_a connects to c");
        $display("  6th port of mod_a connects to d");
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else if (num_tests_passed != 0)
            $display("Overall Result: SOME TESTS PASSED ⚠");
        else
            $display("Overall Result: NO TESTS PASSED ✗");
        $display("===============================================================");
        
        $finish;
    end
    
    // Monitor to track any changes
    initial begin
        $monitor("Time=%0t: a=%b b=%b c=%b d=%b => out1=%b out2=%b", 
            $time, a, b, c, d, out1, out2);
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("top_module_pos_tb.vcd");
        $dumpvars(0, tb_top_module_pos);
        // Also dump internal signals
        $dumpvars(1, dut.instance1);
    end
    
endmodule