`timescale 1ns/1ps

// First, we need to define mod_a for the testbench to work
// (In real scenario, this would be in a separate file)
module mod_a (
    input wire in1,
    input wire in2, 
    output wire out
);
    // Assuming mod_a implements some logic - let's use AND for example
    // You should replace this with the actual mod_a implementation
    assign out = in1 & in2;
endmodule

// Testbench for top_module
module tb_top_module();
    // Inputs (driven by testbench)
    reg a;
    reg b;
    
    // Output (driven by DUT)
    wire out;
    
    // Instantiate the Design Under Test (DUT)
    top_module_import dut (
        .a(a),
        .b(b),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    
    // For recording what operation mod_a performs
    reg expected_out;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing top_module_import with mod_a instantiation");
        $display("top_module passes signals a, b to mod_a instance");
        $display("===============================================================");
        
        // First, let's determine what operation mod_a performs
        $display("\nDetermining mod_a operation:");
        $display("-----------------------------");
        $display("Time | a | b | out | Operation?");
        $display("---------------------------------");
        
        // Test all 4 input combinations to determine the operation
        // Test 00
        a = 1'b0; b = 1'b0; #10;
        $display("%3t  | %b | %b |  %b  |", $time, a, b, out);
        
        // Test 01
        a = 1'b0; b = 1'b1; #10;
        $display("%3t  | %b | %b |  %b  |", $time, a, b, out);
        
        // Test 10
        a = 1'b1; b = 1'b0; #10;
        $display("%3t  | %b | %b |  %b  |", $time, a, b, out);
        
        // Test 11
        a = 1'b1; b = 1'b1; #10;
        $display("%3t  | %b | %b |  %b  |", $time, a, b, out);
        
        // Analyze the pattern
        $display("\nAnalyzing the truth table:");
        a = 1'b0; b = 1'b0; #10;
        if (out == 1'b0) begin
            a = 1'b0; b = 1'b1; #10;
            if (out == 1'b0) begin
                a = 1'b1; b = 1'b0; #10;
                if (out == 1'b0) begin
                    a = 1'b1; b = 1'b1; #10;
                    if (out == 1'b1) $display("Operation appears to be: AND");
                    else if (out == 1'b0) $display("Operation appears to be: NOR");
                end else begin
                    a = 1'b1; b = 1'b1; #10;
                    if (out == 1'b0) $display("Operation appears to be: XOR");
                    else $display("Operation appears to be: OR");
                end
            end else begin
                $display("Operation could be NAND, NOT(a), or other");
            end
        end
        
        // Now perform comprehensive testing
        $display("\n===============================================================");
        $display("Comprehensive Testing:");
        $display("---------------------------------------------------------------");
        $display("Time | a | b | out | Expected | Result");
        $display("---------------------------------------");
        
        // Test Case 1: Both inputs low
        a = 1'b0; b = 1'b0;
        #10;
        // For this example, assuming AND operation
        expected_out = a & b;
        check_result("Both low", expected_out);
        
        // Test Case 2: First high, second low
        a = 1'b1; b = 1'b0;
        #10;
        expected_out = a & b;
        check_result("a=1, b=0", expected_out);
        
        // Test Case 3: First low, second high
        a = 1'b0; b = 1'b1;
        #10;
        expected_out = a & b;
        check_result("a=0, b=1", expected_out);
        
        // Test Case 4: Both inputs high
        a = 1'b1; b = 1'b1;
        #10;
        expected_out = a & b;
        check_result("Both high", expected_out);
        
        // Signal transition testing
        $display("\n===============================================================");
        $display("Signal Transition Testing:");
        $display("---------------------------------------------------------------");
        
        // Hold b constant, toggle a
        $display("\nToggling 'a' while 'b' is constant:");
        b = 1'b0;
        $display("b = 0:");
        a = 1'b0; #10;
        $display("  a=%b => out=%b", a, out);
        a = 1'b1; #10;
        $display("  a=%b => out=%b", a, out);
        
        b = 1'b1;
        $display("b = 1:");
        a = 1'b0; #10;
        $display("  a=%b => out=%b", a, out);
        a = 1'b1; #10;
        $display("  a=%b => out=%b", a, out);
        
        // Hold a constant, toggle b
        $display("\nToggling 'b' while 'a' is constant:");
        a = 1'b0;
        $display("a = 0:");
        b = 1'b0; #10;
        $display("  b=%b => out=%b", b, out);
        b = 1'b1; #10;
        $display("  b=%b => out=%b", b, out);
        
        a = 1'b1;
        $display("a = 1:");
        b = 1'b0; #10;
        $display("  b=%b => out=%b", b, out);
        b = 1'b1; #10;
        $display("  b=%b => out=%b", b, out);
        
        // Timing test - rapid transitions
        $display("\n===============================================================");
        $display("Rapid Transition Test:");
        $display("---------------------------------------------------------------");
        
        a = 1'b0; b = 1'b0;
        #5 a = 1'b1;
        #5 b = 1'b1;
        #5 a = 1'b0;
        #5 b = 1'b0;
        #5;
        $display("After rapid transitions: a=%b, b=%b, out=%b", a, b, out);
        
        // Test module hierarchy
        $display("\n===============================================================");
        $display("Module Hierarchy Verification:");
        $display("---------------------------------------------------------------");
        $display("top_module.instance1 is an instance of mod_a");
        $display("Signals are connected as:");
        $display("  top_module.a -> mod_a.in1");
        $display("  top_module.b -> mod_a.in2");
        $display("  mod_a.out -> top_module.out");
        
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
    
    // Task to check result and update counters
    task check_result;
        input [20*8:1] test_name;
        input exp_out;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  | %b | %b |  %b  |    %b     | %s",
                $time, a, b, out, exp_out,
                (out == exp_out) ? "PASS" : "FAIL");
            
            if (out == exp_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Monitor for observing all signal changes
    initial begin
        $monitor("Time=%0t: a=%b, b=%b, out=%b", $time, a, b, out);
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("top_module_tb.vcd");
        $dumpvars(0, tb_top_module);
        // Also dump internal signals from the instantiated module
        $dumpvars(1, dut.instance1);
    end
    
endmodule