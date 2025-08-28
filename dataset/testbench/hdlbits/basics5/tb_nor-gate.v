`timescale 1ns/1ps

module tb_and_gate();
    // Inputs (driven by testbench)
    reg a, b;
    
    // Outputs (driven by DUT)
    wire out;
    
    // Instantiate the Design Under Test (DUT)
    nor_gate dut (
        .a(a),
        .b(b),
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
        $display("Testing nor_gate module");
        $display("=========================================");
        $display("Time | a | b | Expected | Output | Result");
        $display("-----------------------------------------");

        // Perform NOR GATE Truth Table
        // Test 1: Input ~(0 | 0) = 1
        a = 1'b0;
        b = 1'b0;
        #10;
        if (out == 1'b1) begin
            $display("%3t  | %b | %b |    %b     |   %b    | PASS", 
                $time, a, b, 1'b1, out);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%3t  | %b | %b |    %b     |   %b    | FAIL", 
                $time, a, b, 1'b1, out);
        end
        total_tests = total_tests + 1;

        // Test 2: Input ~(0 | 1) = 0
        a = 1'b0;
        b = 1'b1;
        #10;
        if (out == 1'b0) begin
            $display("%3t  | %b | %b |    %b     |   %b    | PASS", 
                $time, a, b, 1'b0, out);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%3t  | %b | %b |    %b     |   %b    | FAIL", 
                $time, a, b, 1'b0, out);
        end
        total_tests = total_tests + 1;

        // Test 3: Input ~(1 | 0) = 0
        a = 1'b1;
        b = 1'b0;
        #10;
        if (out == 1'b0) begin
            $display("%3t  | %b | %b |    %b     |   %b    | PASS", 
                $time, a, b, 1'b0, out);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%3t  | %b | %b |    %b     |   %b    | FAIL", 
                $time, a, b, 1'b0, out);
        end
        total_tests = total_tests + 1;

        // Test 4: Input ~(1 | 1) = 0
        a = 1'b1;
        b = 1'b1;
        #10;
        if (out == 1'b0) begin
            $display("%3t  | %b | %b |    %b     |   %b    | PASS", 
                $time, a, b, 1'b0, out);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%3t  | %b | %b |    %b     |   %b    | FAIL", 
                $time, a, b, 1'b0, out);
        end
        total_tests = total_tests + 1;

        // Final Summary
        $display("\n=========================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else if (num_tests_passed != 1'b0)
            $display("Overall Result: SOME TESTS PASSED ⚠");
        else
            $display("Overall Result: NO TESTS PASSED ✗");
        $display("=========================================");
        
        $finish;
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("nor_gate_tb.vcd");
        $dumpvars(0, tb_and_gate);
    end
    
endmodule