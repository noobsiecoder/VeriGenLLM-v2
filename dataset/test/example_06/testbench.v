`timescale 1ns / 1ps

module tb_half_adder;
    
    // Inputs
    reg a, b;
    
    // Outputs
    wire sum, carry;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    half_adder uut(
        .a(a),
        .b(b),
        .sum(sum),
        .carry(carry)
    );
    
    // Task to check half adder output
    task check_half_adder;
        input test_a, test_b;
        input expected_sum, expected_carry;
        input [127:0] test_description;
        begin
            a = test_a;
            b = test_b;
            #10; // Wait for propagation
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Inputs: a=%b, b=%b", a, b);
            $display("  Expected: sum=%b, carry=%b", expected_sum, expected_carry);
            $display("  Got:      sum=%b, carry=%b", sum, carry);
            
            if (sum === expected_sum && carry === expected_carry) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                if (sum !== expected_sum)
                    $display("    Sum incorrect: expected %b, got %b", expected_sum, sum);
                if (carry !== expected_carry)
                    $display("    Carry incorrect: expected %b, got %b", expected_carry, carry);
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 6: Half Adder");
        $display("Description: Adds two single bits");
        $display("Truth table:");
        $display("  a | b | sum | carry");
        $display("  0 | 0 |  0  |   0");
        $display("  0 | 1 |  1  |   0");
        $display("  1 | 0 |  1  |   0");
        $display("  1 | 1 |  0  |   1");
        $display("====================================================\n");
        
        // Test all possible input combinations
        $display("Running exhaustive truth table tests:\n");
        
        // Test 1: 0 + 0 = 0 (sum=0, carry=0)
        check_half_adder(1'b0, 1'b0, 1'b0, 1'b0, "0 + 0 = 00");
        
        // Test 2: 0 + 1 = 1 (sum=1, carry=0)
        check_half_adder(1'b0, 1'b1, 1'b1, 1'b0, "0 + 1 = 01");
        
        // Test 3: 1 + 0 = 1 (sum=1, carry=0)
        check_half_adder(1'b1, 1'b0, 1'b1, 1'b0, "1 + 0 = 01");
        
        // Test 4: 1 + 1 = 2 (sum=0, carry=1)
        check_half_adder(1'b1, 1'b1, 1'b0, 1'b1, "1 + 1 = 10");
        
        // Dynamic test - changing inputs
        $display("Dynamic test - verifying output changes:\n");
        
        // Start with both inputs low
        a = 1'b0;
        b = 1'b0;
        #10;
        test_count = test_count + 1;
        $display("Test %0d: Initial state", test_count);
        $display("  a=0, b=0 -> sum=%b, carry=%b", sum, carry);
        if (sum === 1'b0 && carry === 1'b0) begin
            pass_count = pass_count + 1;
            $display("  PASS");
        end else begin
            test_passed = 1'b0;
            $display("  FAIL");
        end
        
        // Change to create sum without carry
        a = 1'b1;
        #10;
        test_count = test_count + 1;
        $display("\nTest %0d: After setting a=1", test_count);
        $display("  a=1, b=0 -> sum=%b, carry=%b", sum, carry);
        if (sum === 1'b1 && carry === 1'b0) begin
            pass_count = pass_count + 1;
            $display("  PASS");
        end else begin
            test_passed = 1'b0;
            $display("  FAIL");
        end
        
        // Create carry
        b = 1'b1;
        #10;
        test_count = test_count + 1;
        $display("\nTest %0d: After setting b=1", test_count);
        $display("  a=1, b=1 -> sum=%b, carry=%b", sum, carry);
        if (sum === 1'b0 && carry === 1'b1) begin
            pass_count = pass_count + 1;
            $display("  PASS");
        end else begin
            test_passed = 1'b0;
            $display("  FAIL");
        end
        
        // Test with unknown values
        $display("\nEdge case tests:\n");
        
        test_count = test_count + 1;
        a = 1'bx;
        b = 1'b0;
        #10;
        $display("Test %0d: Unknown input (a=x, b=0)", test_count);
        $display("  sum=%b, carry=%b", sum, carry);
        if (sum === 1'bx && carry === 1'b0) begin
            $display("  PASS: Unknown propagated to sum, carry is 0");
            pass_count = pass_count + 1;
        end else begin
            $display("  WARNING: Unexpected behavior with unknown input");
            // Don't fail on edge case
        end
        
        // Display test summary
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed && pass_count >= 7) begin  // At least main tests + dynamic tests
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