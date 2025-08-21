`timescale 1ns/1ps

module tb_not_gate();
    // Inputs (driven by testbench)
    reg in;
    
    // Outputs (driven by DUT)
    wire out;
    
    // Instantiate the Design Under Test (DUT)
    not_gate dut (
        .in(in),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("=========================================");
        $display("Testing not_gate module");
        $display("=========================================");
        $display("Time | Input | Expected | Output | Result");
        $display("-----------------------------------------");
        
        // Test both possible input values (0 and 1)
        // Test 1: Input = 0
        in = 1'b0;
        #10;
        if (out == 1'b1) begin
            $display("%3t  |   %b   |    %b     |   %b    | PASS", 
                     $time, in, ~in, out);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%3t  |   %b   |    %b     |   %b    | FAIL", 
                     $time, in, ~in, out);
        end
        total_tests = total_tests + 1;
        
        // Test 2: Input = 1
        in = 1'b1;
        #10;
        if (out == 1'b0) begin
            $display("%3t  |   %b   |    %b     |   %b    | PASS", 
                     $time, in, ~in, out);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%3t  |   %b   |    %b     |   %b    | FAIL", 
                     $time, in, ~in, out);
        end
        total_tests = total_tests + 1;
        
        // Toggle test
        $display("\nToggle test:");
        repeat(10) begin
            in = ~in;  // Toggle input
            #5;
            
            if (out == ~in) begin
                $display("%3t  |   %b   |    %b     |   %b    | PASS", 
                         $time, in, ~in, out);
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("%3t  |   %b   |    %b     |   %b    | FAIL", 
                         $time, in, ~in, out);
            end
            total_tests = total_tests + 1;
        end
        
        // Final Summary
        $display("\n=========================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else
            $display("Overall Result: SOME TESTS FAILED ✗");
        $display("=========================================");
        
        $finish;
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("not_gate_tb.vcd");
        $dumpvars(0, tb_not_gate);
    end
    
endmodule