`timescale 1ns/1ps

module tb_top_module_cond();
    // Inputs
    reg [7:0] a, b, c, d;
    
    // Output
    wire [7:0] min;
    
    // Instantiate DUT
    top_module_cond dut (
        .a(a), .b(b), .c(c), .d(d),
        .min(min)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [7:0] expected_min;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 4-way Minimum Circuit");
        $display("Finding minimum of four 8-bit unsigned numbers");
        $display("===============================================================");
        
        // Test 1: Basic test cases
        $display("\nTest 1: Basic Test Cases");
        $display("---------------------------------------------------------------");
        $display("  a  |  b  |  c  |  d  | min | Expected | Result");
        $display("---------------------------------------------------------------");
        
        // All same
        test_min(8'd50, 8'd50, 8'd50, 8'd50, "All same");
        
        // Min is a
        test_min(8'd10, 8'd20, 8'd30, 8'd40, "Min is a");
        
        // Min is b
        test_min(8'd20, 8'd10, 8'd30, 8'd40, "Min is b");
        
        // Min is c
        test_min(8'd30, 8'd40, 8'd10, 8'd20, "Min is c");
        
        // Min is d
        test_min(8'd40, 8'd30, 8'd20, 8'd10, "Min is d");
        
        // Test 2: Boundary values
        $display("\nTest 2: Boundary Value Tests");
        $display("---------------------------------------------------------------");
        
        // Test with 0
        test_min(8'd0, 8'd100, 8'd200, 8'd50, "Min is 0");
        test_min(8'd100, 8'd0, 8'd200, 8'd50, "Min is 0 (b)");
        
        // Test with 255
        test_min(8'd255, 8'd255, 8'd255, 8'd254, "Min is 254");
        test_min(8'd1, 8'd2, 8'd3, 8'd255, "Max in d position");
        
        // Test 3: Multiple same minimum values
        $display("\nTest 3: Multiple Same Minimum Values");
        $display("---------------------------------------------------------------");
        
        test_min(8'd5, 8'd5, 8'd10, 8'd15, "a and b both min");
        test_min(8'd10, 8'd5, 8'd5, 8'd15, "b and c both min");
        test_min(8'd5, 8'd5, 8'd5, 8'd10, "Three values same");
        
        // Test 4: Sequential values
        $display("\nTest 4: Sequential Values");
        $display("---------------------------------------------------------------");
        
        test_min(8'd1, 8'd2, 8'd3, 8'd4, "Sequential ascending");
        test_min(8'd4, 8'd3, 8'd2, 8'd1, "Sequential descending");
        test_min(8'd2, 8'd4, 8'd1, 8'd3, "Mixed sequential");
        
        // Test 5: Random test cases
        $display("\nTest 5: Random Test Cases");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 10; i = i + 1) begin
            a = $random & 8'hFF;
            b = $random & 8'hFF;
            c = $random & 8'hFF;
            d = $random & 8'hFF;
            
            // Calculate expected minimum
            expected_min = a;
            if (b < expected_min) expected_min = b;
            if (c < expected_min) expected_min = c;
            if (d < expected_min) expected_min = d;
            
            #10;
            
            total_tests = total_tests + 1;
            if (min == expected_min) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%3d | %3d | %3d | %3d | %3d |   %3d    | PASS",
                    a, b, c, d, min, expected_min);
            end else begin
                $display("%3d | %3d | %3d | %3d | %3d |   %3d    | FAIL",
                    a, b, c, d, min, expected_min);
            end
        end
        
        // Test 6: Verify internal structure
        $display("\nTest 6: Internal Structure Verification");
        $display("---------------------------------------------------------------");
        
        a = 8'd40; b = 8'd30; c = 8'd20; d = 8'd10;
        #10;
        
        $display("Inputs: a=%d, b=%d, c=%d, d=%d", a, b, c, d);
        $display("Intermediate values:");
        $display("  min1 = min(a,b) = min(%d,%d) = %d", a, b, dut.min1);
        $display("  min2 = min(c,d) = min(%d,%d) = %d", c, d, dut.min2);
        $display("  Final min = %d", min);
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else
            $display("Overall Result: SOME TESTS PASSED ⚠");
        $display("===============================================================");
        
        $finish;
    end
    
    // Task to test minimum function
    task test_min;
        input [7:0] test_a, test_b, test_c, test_d;
        input [30*8:1] description;
        reg [7:0] exp_min;
        begin
            a = test_a;
            b = test_b;
            c = test_c;
            d = test_d;
            
            // Calculate expected
            exp_min = test_a;
            if (test_b < exp_min) exp_min = test_b;
            if (test_c < exp_min) exp_min = test_c;
            if (test_d < exp_min) exp_min = test_d;
            
            #10;
            
            total_tests = total_tests + 1;
            if (min == exp_min) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%3d | %3d | %3d | %3d | %3d |   %3d    | PASS (%0s)",
                    a, b, c, d, min, exp_min, description);
            end else begin
                $display("%3d | %3d | %3d | %3d | %3d |   %3d    | FAIL (%0s)",
                    a, b, c, d, min, exp_min, description);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("min4_tb.vcd");
        $dumpvars(0, tb_top_module_cond);
    end
    
endmodule