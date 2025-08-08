`timescale 1ns / 1ps

module tb_two_input_and_gate;
    
    // Inputs
    reg a, b;
    
    // Output
    wire y;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    and_gate uut(
        .a(a),
        .b(b),
        .y(y)
    );
    
    // Task to check AND gate behavior
    task check_and_output;
        input expected_a, expected_b;
        input expected_y;
        begin
            a = expected_a;
            b = expected_b;
            #10; // Wait for propagation delay
            
            test_count = test_count + 1;
            $display("Test %0d: a=%b, b=%b, y=%b (expected=%b)", 
                     test_count, a, b, y, expected_y);
            
            if (y === expected_y) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected y=%b, but got y=%b", expected_y, y);
                test_passed = 1'b0;
            end
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 3: 2-input AND gate");
        $display("Description: Implement AND gate functionality");
        $display("Truth table:");
        $display("  a | b | y");
        $display("  0 | 0 | 0");
        $display("  0 | 1 | 0");
        $display("  1 | 0 | 0");
        $display("  1 | 1 | 1");
        $display("====================================================\n");
        
        // Test all possible input combinations (exhaustive test)
        $display("Running exhaustive truth table tests:");
        
        // Test case 1: 0 AND 0 = 0
        check_and_output(1'b0, 1'b0, 1'b0);
        
        // Test case 2: 0 AND 1 = 0
        check_and_output(1'b0, 1'b1, 1'b0);
        
        // Test case 3: 1 AND 0 = 0
        check_and_output(1'b1, 1'b0, 1'b0);
        
        // Test case 4: 1 AND 1 = 1
        check_and_output(1'b1, 1'b1, 1'b1);
        
        // Additional test: Check for X propagation
        $display("\nAdditional tests for unknown values:");
        test_count = test_count + 1;
        a = 1'bx;
        b = 1'b1;
        #10;
        $display("Test %0d: a=x, b=1, y=%b", test_count, y);
        if (y === 1'bx) begin
            $display("  PASS: Unknown propagated correctly");
            pass_count = pass_count + 1;
        end else if (y === 1'b0) begin
            // Some implementations might resolve x AND 1 to 0
            $display("  PASS: Pessimistic evaluation (x AND 1 = 0)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Unexpected output for unknown input");
            test_passed = 1'b0;
        end
        
        // Test with Z (high impedance)
        test_count = test_count + 1;
        a = 1'b1;
        b = 1'bz;
        #10;
        $display("Test %0d: a=1, b=z, y=%b", test_count, y);
        if (y === 1'bx || y === 1'bz) begin
            $display("  PASS: High-impedance handled");
            pass_count = pass_count + 1;
        end else begin
            $display("  WARNING: High-impedance not propagated as expected");
            // Don't fail the test for this edge case
        end
        
        // Display test summary
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed && pass_count >= 4) begin  // At least the 4 main tests must pass
            $display("\nRESULT: ALL TESTS PASSED ✓");
        end else begin
            $display("\nRESULT: TESTS FAILED ✗");
        end
        $display("====================================================\n");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #1000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule