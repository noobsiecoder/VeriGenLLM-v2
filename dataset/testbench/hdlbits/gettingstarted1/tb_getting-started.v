`timescale 1ns/1ps

module tb_top_module();
    // Signal to connect to DUT output
    wire one;
    
    // Instantiate the Design Under Test (DUT)
    top_module dut (
        .one(one)
    );
    
    // Test variables
    reg test_passed;
    integer num_tests_passed;
    
    initial begin
        // Initialize test tracking
        test_passed = 1'b1;
        num_tests_passed = 0;
        
        // Display header
        $display("====================================");
        $display("Testing top_module");
        $display("====================================");
        $display("Time\tOutput\tExpected\tResult");
        $display("------------------------------------");
        
        // Wait for any initialization
        #10;
        
        // Test Case 1: Check if 'one' equals plain 1
        if (one == 1) begin
            $display("%0t\t%b\t1\t\tPASS", $time, one);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%0t\t%b\t1\t\tFAIL", $time, one);
            test_passed = 1'b0;
        end
        
        // Test Case 2: Check if 'one' equals 1'b1
        #10;
        if (one == 1'b1) begin
            $display("%0t\t%b\t1'b1\t\tPASS", $time, one);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%0t\t%b\t1'b1\t\tFAIL", $time, one);
            test_passed = 1'b0;
        end
        
        // Test Case 3: Check if 'one' is NOT 1'b0
        #10;
        if (one != 1'b0) begin
            $display("%0t\t%b\t!1'b0\t\tPASS", $time, one);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%0t\t%b\t!1'b0\t\tFAIL", $time, one);
            test_passed = 1'b0;
        end
        
        // Test Case 4: Check exact match with === operator
        #10;
        if (one === 1'b1) begin
            $display("%0t\t%b\t===1'b1\t\tPASS", $time, one);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%0t\t%b\t===1'b1\t\tFAIL", $time, one);
            test_passed = 1'b0;
        end
        
        // Final Summary
        #10;
        $display("====================================");
        $display("Test Summary: %0d/4 tests passed", num_tests_passed);
        if (test_passed)
            $display("Overall Result: ALL TESTS PASSED");
        else
            $display("Overall Result: SOME TESTS FAILED");
        $display("====================================");
        
        // End simulation
        #10;
        $finish;
    end
    
    // Optional: Generate VCD file for waveform viewing
    initial begin
        $dumpfile("top_module_tb.vcd");
        $dumpvars(0, tb_top_module);
    end
    
endmodule