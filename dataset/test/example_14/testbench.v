`timescale 1ns / 1ps

module tb_signed_adder_8bit;
    
    // Inputs
    reg signed [7:0] a;
    reg signed [7:0] b;
    
    // Outputs
    wire signed [7:0] sum;
    wire overflow;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // For calculating expected values
    reg signed [8:0] expected_sum;  // 9-bit to detect overflow
    reg expected_overflow;
    
    // Instantiate the module under test
    signed_adder_8bit uut(
        .a(a),
        .b(b),
        .sum(sum),
        .overflow(overflow)
    );
    
    // Task to check adder output
    task check_addition;
        input signed [7:0] test_a;
        input signed [7:0] test_b;
        input [127:0] test_description;
        begin
            a = test_a;
            b = test_b;
            #10; // Wait for combinational logic
            
            // Calculate expected values
            expected_sum = test_a + test_b;
            
            // Overflow detection for signed addition:
            // Overflow if both operands have same sign but result has different sign
            expected_overflow = (~test_a[7] & ~test_b[7] & expected_sum[7]) |  // pos + pos = neg
                               ( test_a[7] &  test_b[7] & ~expected_sum[7]);   // neg + neg = pos
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Inputs: a=%d (%h), b=%d (%h)", $signed(test_a), test_a, $signed(test_b), test_b);
            $display("  Expected: sum=%d (%h), overflow=%b", $signed(expected_sum[7:0]), expected_sum[7:0], expected_overflow);
            $display("  Got:      sum=%d (%h), overflow=%b", $signed(sum), sum, overflow);
            
            if (sum === expected_sum[7:0] && overflow === expected_overflow) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                if (sum !== expected_sum[7:0])
                    $display("    Sum incorrect");
                if (overflow !== expected_overflow)
                    $display("    Overflow flag incorrect");
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 14: Signed 8-bit Adder with Overflow Detection");
        $display("Description: Add two 8-bit signed numbers (2's complement)");
        $display("Overflow occurs when:");
        $display("  - Positive + Positive = Negative");
        $display("  - Negative + Negative = Positive");
        $display("====================================================\n");
        
        // Test 1: Basic additions without overflow
        $display("Test Set 1: Basic additions (no overflow)");
        check_addition(8'd10, 8'd20, "10 + 20 = 30");
        check_addition(-8'd10, -8'd20, "-10 + (-20) = -30");
        check_addition(8'd50, -8'd30, "50 + (-30) = 20");
        check_addition(-8'd50, 8'd30, "-50 + 30 = -20");
        check_addition(8'd0, 8'd0, "0 + 0 = 0");
        
        // Test 2: Boundary cases without overflow
        $display("Test Set 2: Boundary cases (no overflow)");
        check_addition(8'h7F, 8'h00, "127 + 0 = 127 (max positive)");
        check_addition(8'h80, 8'h00, "-128 + 0 = -128 (min negative)");
        check_addition(8'h01, 8'hFF, "1 + (-1) = 0");
        check_addition(8'd64, 8'd63, "64 + 63 = 127");
        check_addition(-8'd64, -8'd64, "-64 + (-64) = -128");
        
        // Test 3: Positive overflow cases
        $display("Test Set 3: Positive overflow cases");
        check_addition(8'h7F, 8'h01, "127 + 1 = 128 (overflow)");
        check_addition(8'h7F, 8'h7F, "127 + 127 = 254 (overflow)");
        check_addition(8'd100, 8'd50, "100 + 50 = 150 (overflow)");
        check_addition(8'd64, 8'd64, "64 + 64 = 128 (overflow)");
        
        // Test 4: Negative overflow cases
        $display("Test Set 4: Negative overflow cases");
        check_addition(8'h80, 8'hFF, "-128 + (-1) = -129 (overflow)");
        check_addition(8'h80, 8'h80, "-128 + (-128) = -256 (overflow)");
        check_addition(-8'd100, -8'd50, "-100 + (-50) = -150 (overflow)");
        check_addition(-8'd65, -8'd65, "-65 + (-65) = -130 (overflow)");
        
        // Test 5: Mixed sign additions (never overflow)
        $display("Test Set 5: Mixed sign additions (no overflow possible)");
        check_addition(8'h7F, 8'h80, "127 + (-128) = -1");
        check_addition(8'h80, 8'h7F, "-128 + 127 = -1");
        check_addition(8'd100, -8'd100, "100 + (-100) = 0");
        check_addition(-8'd75, 8'd75, "-75 + 75 = 0");
        
        // Test 6: Edge cases around overflow boundary
        $display("Test Set 6: Edge cases near overflow");
        check_addition(8'd126, 8'd1, "126 + 1 = 127 (no overflow)");
        check_addition(-8'd127, -8'd1, "-127 + (-1) = -128 (no overflow)");
        check_addition(8'd63, 8'd64, "63 + 64 = 127 (no overflow)");
        check_addition(-8'd63, -8'd65, "-63 + (-65) = -128 (no overflow)");
        
        // Test 7: Verify combinational behavior
        $display("Test Set 7: Combinational behavior");
        a = 8'd50;
        b = 8'd50;
        #5;
        if (sum === 8'd100 && overflow === 1'b0) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Combinational output correct", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Combinational output incorrect", test_count);
            test_passed = 1'b0;
        end
        
        // Change to overflow condition
        a = 8'd100;
        b = 8'd100;
        #5;
        if ((sum == 8'hC8 || sum == -8'd56) && overflow === 1'b1) begin  // 200 in 8-bit = -56 or 0xC8
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Overflow detected correctly", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Overflow not detected", test_count);
            test_passed = 1'b0;
        end
        
        // Test 8: Special bit patterns
        $display("\nTest Set 8: Special bit patterns");
        check_addition(8'b01111111, 8'b00000001, "01111111 + 00000001 (127 + 1)");
        check_addition(8'b10000000, 8'b11111111, "10000000 + 11111111 (-128 + -1)");
        check_addition(8'b01010101, 8'b00101010, "01010101 + 00101010 (85 + 42)");
        
        // Display test summary
        #10;
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed) begin
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