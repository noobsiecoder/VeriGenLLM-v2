`timescale 1ns/1ps

// First, we need to define mod_a for the testbench to work
// (In real scenario, this would be in a separate file)
module mod_a (
    input wire in1,
    input wire in2,
    input wire in3,
    input wire in4,
    output wire out1,
    output wire out2
);
    // Example implementation - replace with actual logic
    assign out1 = (in1 & in2) | (in3 & in4);  // Example: OR of two AND operations
    assign out2 = (in1 ^ in2) ^ (in3 ^ in4);  // Example: XOR of all inputs
endmodule

// Testbench for top_module_name
module tb_top_module_name();
    // Inputs (driven by testbench)
    reg a, b, c, d;
    
    // Outputs (driven by DUT)
    wire out1, out2;
    
    // Instantiate the Design Under Test (DUT)
    top_module_name dut (
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
    
    // Expected values
    reg expected_out1, expected_out2;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing top_module_name with named port connections");
        $display("Port mapping (named connections):");
        $display("  top_module.a    -> mod_a.in1");
        $display("  top_module.b    -> mod_a.in2");
        $display("  top_module.c    -> mod_a.in3");
        $display("  top_module.d    -> mod_a.in4");
        $display("  mod_a.out1      -> top_module.out1");
        $display("  mod_a.out2      -> top_module.out2");
        $display("===============================================================");
        
        // Test all 16 possible input combinations
        $display("\nComplete Truth Table:");
        $display("---------------------------------------------");
        $display("Index | a b c d | out1 out2 | Binary | Hex");
        $display("---------------------------------------------");
        
        for (i = 0; i < 16; i = i + 1) begin
            {a, b, c, d} = i[3:0];
            #10;
            $display(" %2d   | %b %b %b %b |  %b    %b   | %4b  | %X", 
                i, a, b, c, d, out1, out2, i[3:0], i[3:0]);
        end
        
        // Analyze output patterns
        $display("\n===============================================================");
        $display("Output Pattern Analysis:");
        $display("---------------------------------------------------------------");
        
        // Check for specific patterns
        $display("\nChecking when out1 = 1:");
        for (i = 0; i < 16; i = i + 1) begin
            {a, b, c, d} = i[3:0];
            #10;
            if (out1 == 1'b1) begin
                $display("  a=%b b=%b c=%b d=%b => out1=1", a, b, c, d);
            end
        end
        
        $display("\nChecking when out2 = 1:");
        for (i = 0; i < 16; i = i + 1) begin
            {a, b, c, d} = i[3:0];
            #10;
            if (out2 == 1'b1) begin
                $display("  a=%b b=%b c=%b d=%b => out2=1", a, b, c, d);
            end
        end
        
        // Detailed test cases
        $display("\n===============================================================");
        $display("Detailed Test Cases:");
        $display("---------------------------------------------------------------");
        $display("Time | a b c d | out1 out2 | Expected | Result");
        $display("-----------------------------------------------");
        
        // Test Case 1: All zeros
        {a, b, c, d} = 4'b0000;
        expected_out1 = 1'b0;  // (0&0)|(0&0) = 0
        expected_out2 = 1'b0;  // (0^0)^(0^0) = 0
        #10;
        check_result("All zeros", expected_out1, expected_out2);
        
        // Test Case 2: Individual bits
        {a, b, c, d} = 4'b1000;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("Only a=1", expected_out1, expected_out2);
        
        {a, b, c, d} = 4'b0100;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("Only b=1", expected_out1, expected_out2);
        
        {a, b, c, d} = 4'b0010;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("Only c=1", expected_out1, expected_out2);
        
        {a, b, c, d} = 4'b0001;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("Only d=1", expected_out1, expected_out2);
        
        // Test Case 3: Pairs active
        {a, b, c, d} = 4'b1100;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("a=1,b=1", expected_out1, expected_out2);
        
        {a, b, c, d} = 4'b0011;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("c=1,d=1", expected_out1, expected_out2);
        
        // Test Case 4: All ones
        {a, b, c, d} = 4'b1111;
        expected_out1 = (a & b) | (c & d);
        expected_out2 = (a ^ b) ^ (c ^ d);
        #10;
        check_result("All ones", expected_out1, expected_out2);
        
        // Symmetry testing
        $display("\n===============================================================");
        $display("Symmetry Testing:");
        $display("---------------------------------------------------------------");
        
        // Test if swapping (a,b) with (c,d) gives same out1
        {a, b, c, d} = 4'b1001;
        #10;
        $display("a=%b b=%b c=%b d=%b => out1=%b out2=%b", a, b, c, d, out1, out2);
        
        {a, b, c, d} = 4'b0110;
        #10;
        $display("a=%b b=%b c=%b d=%b => out1=%b out2=%b (swapped)", a, b, c, d, out1, out2);
        
        // Signal propagation delay test
        $display("\n===============================================================");
        $display("Signal Propagation Test:");
        $display("---------------------------------------------------------------");
        
        {a, b, c, d} = 4'b0000;
        #10;
        $display("Initial: a=%b b=%b c=%b d=%b => out1=%b out2=%b", 
            a, b, c, d, out1, out2);
        
        a = 1'b1; #2;
        $display("After 2ns (a=1): out1=%b out2=%b", out1, out2);
        
        b = 1'b1; #2;
        $display("After 4ns (a=1,b=1): out1=%b out2=%b", out1, out2);
        
        c = 1'b1; #2;
        $display("After 6ns (a=1,b=1,c=1): out1=%b out2=%b", out1, out2);
        
        d = 1'b1; #2;
        $display("After 8ns (all 1s): out1=%b out2=%b", out1, out2);
        
        // Verify all tests
        $display("\n===============================================================");
        $display("Complete Verification:");
        $display("---------------------------------------------------------------");
        
        total_tests = 0;
        num_tests_passed = 0;
        
        for (i = 0; i < 16; i = i + 1) begin
            {a, b, c, d} = i[3:0];
            expected_out1 = (a & b) | (c & d);
            expected_out2 = (a ^ b) ^ (c ^ d);
            #10;
            
            total_tests = total_tests + 1;
            if (out1 == expected_out1 && out2 == expected_out2) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
        
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
    
    // Task to check results
    task check_result;
        input [20*8:1] test_name;
        input exp_out1, exp_out2;
        begin
            $display("%3t  | %b %b %b %b |  %b    %b   | %b    %b   | %s",
                $time, a, b, c, d, out1, out2, exp_out1, exp_out2,
                (out1 == exp_out1 && out2 == exp_out2) ? "PASS" : "FAIL");
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("top_module_name_tb.vcd");
        $dumpvars(0, tb_top_module_name);
        // Also dump the instantiated module's signals
        $dumpvars(1, dut.instance1);
    end
    
endmodule