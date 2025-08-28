`timescale 1ns/1ps

module tb_top_module_gates100();
    // Input
    reg [99:0] in;
    
    // Outputs
    wire out_and;
    wire out_or;
    wire out_xor;
    
    // Instantiate DUT
    top_module_gates100 dut (
        .in(in),
        .out_and(out_and),
        .out_or(out_or),
        .out_xor(out_xor)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    reg expected_and, expected_or, expected_xor;
    integer ones_count;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 100-input Gates");
        $display("AND: All bits must be 1");
        $display("OR: At least one bit must be 1");
        $display("XOR: Odd number of 1s");
        $display("===============================================================");
        
        // Test 1: Basic cases
        $display("\nTest 1: Basic Test Cases");
        $display("---------------------------------------------------------------");
        
        // All zeros
        in = 100'h0;
        #10;
        check_gates("All zeros", 1'b0, 1'b0, 1'b0);
        
        // All ones
        in = {100{1'b1}};
        #10;
        check_gates("All ones", 1'b1, 1'b1, 1'b0);
        
        // Single bit set
        in = 100'h1;
        #10;
        check_gates("Single bit (LSB)", 1'b0, 1'b1, 1'b1);
        
        in = 100'h1 << 99;
        #10;
        check_gates("Single bit (MSB)", 1'b0, 1'b1, 1'b1);
        
        // Test 2: Pattern tests
        $display("\nTest 2: Pattern Tests");
        $display("---------------------------------------------------------------");
        
        // Alternating bits (50 ones)
        in = 100'h5555555555555555555555555;
        #10;
        ones_count = 50;
        check_gates("Alternating (50 ones)", 1'b0, 1'b1, 1'b0);
        
        // First 50 bits set
        in = {50'h0, {50{1'b1}}};
        #10;
        check_gates("Lower 50 bits", 1'b0, 1'b1, 1'b0);
        
        // Test 3: AND gate verification
        $display("\nTest 3: AND Gate Tests");
        $display("---------------------------------------------------------------");
        
        // All ones except one bit
        in = {100{1'b1}};
        in[50] = 1'b0;  // Clear middle bit
        #10;
        $display("All 1s except bit 50: AND=%b (should be 0)", out_and);
        check_result(out_and == 1'b0, "AND with one 0");
        
        // Restore to all ones
        in[50] = 1'b1;
        #10;
        $display("All 1s restored: AND=%b (should be 1)", out_and);
        check_result(out_and == 1'b1, "AND all 1s");
        
        // Test 4: OR gate verification
        $display("\nTest 4: OR Gate Tests");
        $display("---------------------------------------------------------------");
        
        // Start with all zeros
        in = 100'h0;
        #10;
        $display("All 0s: OR=%b (should be 0)", out_or);
        check_result(out_or == 1'b0, "OR all 0s");
        
        // Set just one bit
        in[75] = 1'b1;
        #10;
        $display("Single bit 75 set: OR=%b (should be 1)", out_or);
        check_result(out_or == 1'b1, "OR with one 1");
        
        // Test 5: XOR parity tests
        $display("\nTest 5: XOR Parity Tests");
        $display("---------------------------------------------------------------");
        
        // Test different numbers of ones
        for (i = 0; i <= 10; i = i + 1) begin
            in = 100'h0;
            // Set i bits to 1
            for (j = 0; j < i; j = j + 1) begin
                in[j] = 1'b1;
            end
            #10;
            
            expected_xor = i[0];  // LSB determines odd/even
            $display("%2d ones: XOR=%b (expect %b) %s", 
                i, out_xor, expected_xor,
                (out_xor == expected_xor) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (out_xor == expected_xor) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
        
        // Test 6: Random patterns
        $display("\nTest 6: Random Pattern Tests");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 10; i = i + 1) begin
            // Generate random 100-bit pattern
            in[31:0] = $random;
            in[63:32] = $random;
            in[95:64] = $random;
            in[99:96] = $random & 4'hF;
            
            #10;
            
            // Count ones for XOR check
            ones_count = 0;
            for (j = 0; j < 100; j = j + 1) begin
                if (in[j]) ones_count = ones_count + 1;
            end
            
            // Calculate expected values
            expected_and = (ones_count == 100) ? 1'b1 : 1'b0;
            expected_or = (ones_count > 0) ? 1'b1 : 1'b0;
            expected_xor = ones_count[0];
            
            total_tests = total_tests + 3;
            
            $display("Random %2d: ones=%3d, AND=%b(%b), OR=%b(%b), XOR=%b(%b) %s",
                i, ones_count, 
                out_and, expected_and,
                out_or, expected_or,
                out_xor, expected_xor,
                ((out_and == expected_and) && (out_or == expected_or) && 
                 (out_xor == expected_xor)) ? "PASS" : "FAIL");
                 
            if (out_and == expected_and) num_tests_passed = num_tests_passed + 1;
            if (out_or == expected_or) num_tests_passed = num_tests_passed + 1;
            if (out_xor == expected_xor) num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 7: Edge cases
        $display("\nTest 7: Edge Cases");
        $display("---------------------------------------------------------------");
        
        // 99 ones (even)
        in = {100{1'b1}};
        in[0] = 1'b0;
        #10;
        check_gates("99 ones", 1'b0, 1'b1, 1'b1);
        
        // 1 zero in different positions
        for (i = 0; i < 100; i = i + 10) begin
            in = {100{1'b1}};
            in[i] = 1'b0;
            #10;
            
            $display("Zero at bit %2d: AND=%b, OR=%b, XOR=%b", 
                i, out_and, out_or, out_xor);
        end
        
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
    
    // Task to check all three gates
    task check_gates;
        input [30*8:1] description;
        input exp_and, exp_or, exp_xor;
        begin
            total_tests = total_tests + 3;
            
            $display("%0s:", description);
            $display("  AND: %b (expect %b) %s", out_and, exp_and,
                (out_and == exp_and) ? "PASS" : "FAIL");
            $display("  OR:  %b (expect %b) %s", out_or, exp_or,
                (out_or == exp_or) ? "PASS" : "FAIL");
            $display("  XOR: %b (expect %b) %s", out_xor, exp_xor,
                (out_xor == exp_xor) ? "PASS" : "FAIL");
                
            if (out_and == exp_and) num_tests_passed = num_tests_passed + 1;
            if (out_or == exp_or) num_tests_passed = num_tests_passed + 1;
            if (out_xor == exp_xor) num_tests_passed = num_tests_passed + 1;
        end
    endtask
    
    // Task to check single result
    task check_result;
        input condition;
        input [30*8:1] description;
        begin
            total_tests = total_tests + 1;
            if (condition) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  %0s: PASS", description);
            end else begin
                $display("  %0s: FAIL", description);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("gates100_tb.vcd");
        $dumpvars(0, tb_top_module_gates100);
    end
    
endmodule