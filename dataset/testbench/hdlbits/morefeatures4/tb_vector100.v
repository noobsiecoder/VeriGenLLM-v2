`timescale 1ns/1ps

module tb_top_module_vector100();
    // Input
    reg [99:0] in;
    
    // Output
    wire [99:0] out;
    
    // Instantiate DUT
    top_module_vector100 dut (
        .in(in),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    reg [99:0] expected;
    reg test_passed;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 100-bit Vector Reversal");
        $display("Output should be bit-reversed version of input");
        $display("===============================================================");
        
        // Test 1: Basic cases
        $display("\nTest 1: Basic Test Cases");
        $display("---------------------------------------------------------------");
        
        // All zeros
        in = 100'h0;
        #10;
        expected = 100'h0;
        check_result("All zeros", expected);
        
        // All ones
        in = {100{1'b1}};
        #10;
        expected = {100{1'b1}};
        check_result("All ones", expected);
        
        // Single bit at LSB
        in = 100'h1;
        #10;
        expected = 100'h1 << 99;
        check_result("Single bit at LSB", expected);
        
        // Single bit at MSB
        in = 100'h1 << 99;
        #10;
        expected = 100'h1;
        check_result("Single bit at MSB", expected);
        
        // Test 2: Pattern tests
        $display("\nTest 2: Pattern Tests");
        $display("---------------------------------------------------------------");
        
        // Alternating bits (0101...)
        in = 100'h5555555555555555555555555;
        #10;
        expected = 100'hAAAAAAAAAAAAAAAAAAAAAAAAA;
        check_result("Alternating pattern 0101", expected);
        
        // Alternating bits (1010...)
        in = 100'hAAAAAAAAAAAAAAAAAAAAAAAAA;
        #10;
        expected = 100'h5555555555555555555555555;
        check_result("Alternating pattern 1010", expected);
        
        // First half ones, second half zeros
        in = {50'h0, {50{1'b1}}};
        #10;
        expected = {{50{1'b1}}, 50'h0};
        check_result("Half ones/zeros", expected);
        
        // Test 3: Sequential patterns
        $display("\nTest 3: Sequential Pattern Tests");
        $display("---------------------------------------------------------------");
        
        // Walking one
        for (i = 0; i < 10; i = i + 1) begin
            in = 100'h1 << i;
            #10;
            expected = 100'h1 << (99 - i);
            $display("Walking one at bit %2d: %s", i, 
                (out == expected) ? "PASS" : "FAIL");
            total_tests = total_tests + 1;
            if (out == expected) num_tests_passed = num_tests_passed + 1;
        end
        
        // Walking zero
        for (i = 0; i < 10; i = i + 1) begin
            in = ~(100'h1 << i);
            #10;
            expected = ~(100'h1 << (99 - i));
            $display("Walking zero at bit %2d: %s", i, 
                (out == expected) ? "PASS" : "FAIL");
            total_tests = total_tests + 1;
            if (out == expected) num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 4: Bit position verification
        $display("\nTest 4: Bit Position Verification");
        $display("---------------------------------------------------------------");
        
        // Check specific bit mappings
        in = 100'h0;
        in[0] = 1'b1;
        in[99] = 1'b1;
        in[50] = 1'b1;
        #10;
        
        test_passed = 1'b1;
        if (out[99] != in[0]) test_passed = 1'b0;
        if (out[0] != in[99]) test_passed = 1'b0;
        if (out[49] != in[50]) test_passed = 1'b0;
        
        $display("Bit mapping test: %s", test_passed ? "PASS" : "FAIL");
        $display("  in[0]=%b should map to out[99]=%b", in[0], out[99]);
        $display("  in[99]=%b should map to out[0]=%b", in[99], out[0]);
        $display("  in[50]=%b should map to out[49]=%b", in[50], out[49]);
        
        total_tests = total_tests + 1;
        if (test_passed) num_tests_passed = num_tests_passed + 1;
        
        // Test 5: Random patterns
        $display("\nTest 5: Random Pattern Tests");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 20; i = i + 1) begin
            // Generate random pattern
            in[31:0] = $random;
            in[63:32] = $random;
            in[95:64] = $random;
            in[99:96] = $random & 4'hF;
            
            #10;
            
            // Calculate expected reversal
            for (j = 0; j < 100; j = j + 1) begin
                expected[99-j] = in[j];
            end
            
            test_passed = (out == expected);
            total_tests = total_tests + 1;
            
            $display("Random %2d: %s", i, test_passed ? "PASS" : "FAIL");
            
            if (test_passed) begin
                num_tests_passed = num_tests_passed + 1;
            end else begin
                // Show first mismatch for debugging
                j = 0;
                while (j < 100 && out[j] == expected[j]) begin
                    j = j + 1;
                end
                if (j < 100) begin
                    $display("  First mismatch at bit %2d: out=%b, expected=%b", 
                        j, out[j], expected[j]);
                end
            end
        end
        
        // Test 6: Edge cases
        $display("\nTest 6: Edge Cases");
        $display("---------------------------------------------------------------");
        
        // Palindrome pattern (should equal itself when reversed)
        in = 100'h0;
        for (i = 0; i < 50; i = i + 1) begin
            in[i] = i[0];  // Alternating pattern
            in[99-i] = i[0];  // Mirror it
        end
        #10;
        check_result("Palindrome pattern", in);
        
        // Counting pattern
        in = 100'h0;
        for (i = 0; i < 25; i = i + 1) begin
            in[4*i +: 4] = i[3:0];
        end
        #10;
        
        // Calculate expected for counting pattern
        for (i = 0; i < 100; i = i + 1) begin
            expected[99-i] = in[i];
        end
        check_result("Counting pattern", expected);
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else
            $display("Overall Result: SOME TESTS FAILED ⚠");
        $display("===============================================================");
        
        $finish;
    end
    
    // Task to check result
    task check_result;
        input [50*8:1] description;
        input [99:0] expected_value;
        begin
            total_tests = total_tests + 1;
            
            if (out == expected_value) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%0s: PASS", description);
            end else begin
                $display("%0s: FAIL", description);
                $display("  Expected: %h", expected_value);
                $display("  Got:      %h", out);
                
                // Find first bit difference
                i = 99;
                while (i >= 0 && out[i] == expected_value[i]) begin
                    i = i - 1;
                end
                if (i >= 0) begin
                    $display("  First difference at bit %2d: out=%b, expected=%b", 
                        i, out[i], expected_value[i]);
                end
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("vector100_tb.vcd");
        $dumpvars(0, tb_top_module_vector100);
    end
    
endmodule