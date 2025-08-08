`timescale 1ns / 1ps

module tb_and_truth_table;
    
    // Inputs
    reg a, b;
    
    // Output
    wire y;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    and_gate_case uut(
        .a(a),
        .b(b),
        .y(y)
    );
    
    // Task to check AND gate behavior
    task check_and_output;
        input test_a, test_b;
        input expected_y;
        input [127:0] test_description;
        begin
            a = test_a;
            b = test_b;
            #10; // Wait for combinational logic
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Inputs: a=%b, b=%b", a, b);
            $display("  Expected: y=%b", expected_y);
            $display("  Got:      y=%b", y);
            
            if (y === expected_y) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 13: Truth Table - AND gate using case statement");
        $display("Description: Implement 2-input AND using case statement");
        $display("Expected truth table:");
        $display("  a | b | y");
        $display("  --|---|---");
        $display("  0 | 0 | 0");
        $display("  0 | 1 | 0");
        $display("  1 | 0 | 0");
        $display("  1 | 1 | 1");
        $display("====================================================\n");
        
        // Test all possible input combinations (exhaustive test)
        $display("Testing complete truth table:");
        
        // Test case 1: 00 -> 0
        check_and_output(1'b0, 1'b0, 1'b0, "Case 00: 0 AND 0 = 0");
        
        // Test case 2: 01 -> 0
        check_and_output(1'b0, 1'b1, 1'b0, "Case 01: 0 AND 1 = 0");
        
        // Test case 3: 10 -> 0
        check_and_output(1'b1, 1'b0, 1'b0, "Case 10: 1 AND 0 = 0");
        
        // Test case 4: 11 -> 1
        check_and_output(1'b1, 1'b1, 1'b1, "Case 11: 1 AND 1 = 1");
        
        // Test combinational behavior
        $display("Testing combinational behavior:");
        
        // Set inputs
        a = 1'b0;
        b = 1'b0;
        #5;
        if (y === 1'b0) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output is combinational (00->0)", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Output not combinational", test_count);
            test_passed = 1'b0;
        end
        
        // Change one input
        a = 1'b1;
        #5;
        if (y === 1'b0) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output updates with input change (10->0)", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Output doesn't update correctly", test_count);
            test_passed = 1'b0;
        end
        
        // Change to make output 1
        b = 1'b1;
        #5;
        if (y === 1'b1) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - AND gate produces 1 when both inputs are 1", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - AND gate doesn't produce 1 for 11", test_count);
            test_passed = 1'b0;
        end
        
        // Test rapid input changes
        $display("\nTesting rapid input changes:");
        a = 1'b0; b = 1'b0; #5;
        a = 1'b1; b = 1'b0; #5;
        a = 1'b0; b = 1'b1; #5;
        a = 1'b1; b = 1'b1; #5;
        
        test_count = test_count + 1;
        if (y === 1'b1) begin
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output follows rapid input changes", test_count);
        end else begin
            $display("Test %0d: FAIL - Output doesn't follow rapid changes", test_count);
            test_passed = 1'b0;
        end
        
        // Edge case: Unknown inputs (optional test)
        $display("\nTesting edge cases:");
        a = 1'bx;
        b = 1'b1;
        #10;
        test_count = test_count + 1;
        $display("Test %0d: Edge case - input a=x, b=1", test_count);
        $display("  Output y=%b (x is acceptable for unknown input)", y);
        if (y === 1'bx || y === 1'b0) begin
            pass_count = pass_count + 1;
            $display("  PASS - Handles unknown input reasonably");
        end else if (y === 1'b1) begin
            $display("  WARNING - Output is 1 with unknown input (not typical AND behavior)");
        end
        
        // Test with z (high impedance)
        a = 1'b1;
        b = 1'bz;
        #10;
        test_count = test_count + 1;
        $display("\nTest %0d: Edge case - input a=1, b=z", test_count);
        $display("  Output y=%b", y);
        if (y === 1'bx || y === 1'bz) begin
            pass_count = pass_count + 1;
            $display("  PASS - Handles high-impedance input");
        end else begin
            $display("  INFO - Implementation specific behavior for z input");
        end
        
        // Display test summary
        #10;
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed && pass_count >= 7) begin  // At least main tests must pass
            $display("\nRESULT: ALL TESTS PASSED ✓");
        end else begin
            $display("\nRESULT: TESTS FAILED ✗");
        end
        $display("====================================================\n");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #10000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule