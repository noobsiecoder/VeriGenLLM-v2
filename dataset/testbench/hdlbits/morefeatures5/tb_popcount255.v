`timescale 1ns/1ps

module tb_top_module_popcount255();
    // Input
    reg [254:0] in;
    
    // Output
    wire [7:0] out;
    
    // Instantiate DUT
    top_module_popcount255 dut (
        .in(in),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j, k;
    integer expected_count;
    integer actual_ones;
    reg [7:0] expected;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 255-bit Population Count");
        $display("Output should equal the number of 1's in input");
        $display("===============================================================");
        
        // Test 1: Basic cases
        $display("\nTest 1: Basic Test Cases");
        $display("---------------------------------------------------------------");
        
        // All zeros
        in = 255'h0;
        #10;
        expected = 8'd0;
        check_result("All zeros", expected);
        
        // All ones
        in = {255{1'b1}};
        #10;
        expected = 8'd255;
        check_result("All ones", expected);
        
        // Single bit set - LSB
        in = 255'h1;
        #10;
        expected = 8'd1;
        check_result("Single bit (LSB)", expected);
        
        // Single bit set - MSB
        in = 255'h1 << 254;
        #10;
        expected = 8'd1;
        check_result("Single bit (MSB)", expected);
        
        // Test 2: Pattern tests
        $display("\nTest 2: Pattern Tests");
        $display("---------------------------------------------------------------");
        
        // First 8 bits set
        in = 255'hFF;
        #10;
        expected = 8'd8;
        check_result("First byte set", expected);
        
        // Alternating bits (0101...)
        in = {{127{2'b01}}, 1'b1};  // 127 pairs of 01 plus one 1 = 255 bits total
        #10;
        expected = 8'd128;  // 128 ones
        check_result("Alternating pattern 0101", expected);
        
        // Alternating bits (1010...)
        in = {{127{2'b10}}, 1'b1};  // 127 pairs of 10 plus one 1 = 255 bits total
        #10;
        expected = 8'd128;  // 128 ones
        check_result("Alternating pattern 1010", expected);
        
        // Test 3: Progressive bit counts
        $display("\nTest 3: Progressive Bit Count Tests");
        $display("---------------------------------------------------------------");
        
        // Test counts from 0 to 20
        for (i = 0; i <= 20; i = i + 1) begin
            in = 255'h0;
            // Set i bits to 1
            for (j = 0; j < i; j = j + 1) begin
                in[j] = 1'b1;
            end
            #10;
            
            expected = i;
            $display("Setting %3d bits: out=%3d (expect %3d) %s", 
                i, out, expected,
                (out == expected) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (out == expected) num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 4: Specific counts
        $display("\nTest 4: Specific Count Tests");
        $display("---------------------------------------------------------------");
        
        // Test powers of 2
        for (i = 0; i < 8; i = i + 1) begin
            in = 255'h0;
            // Set 2^i bits
            for (j = 0; j < (1 << i); j = j + 1) begin
                in[j] = 1'b1;
            end
            #10;
            
            expected = 1 << i;
            $display("%3d bits set: out=%3d %s", 
                expected, out,
                (out == expected) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (out == expected) num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 5: Random bit positions
        $display("\nTest 5: Random Bit Position Tests");
        $display("---------------------------------------------------------------");
        
        // Walking multiple bits
        for (i = 1; i <= 5; i = i + 1) begin
            in = 255'h0;
            // Set bits at specific positions
            for (j = 0; j < 255; j = j + (255/i)) begin
                in[j] = 1'b1;
            end
            #10;
            
            // Count actual ones
            actual_ones = 0;
            for (j = 0; j < 255; j = j + 1) begin
                if (in[j]) actual_ones = actual_ones + 1;
            end
            
            expected = actual_ones;
            $display("Spaced bits (gap=%3d): count=%3d %s", 
                255/i, out,
                (out == expected) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (out == expected) num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 6: Random patterns
        $display("\nTest 6: Random Pattern Tests");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 20; i = i + 1) begin
            // Generate random pattern
            in[31:0] = $random;
            in[63:32] = $random;
            in[95:64] = $random;
            in[127:96] = $random;
            in[159:128] = $random;
            in[191:160] = $random;
            in[223:192] = $random;
            in[254:224] = $random & 31'h7FFFFFFF;  // Only 31 bits for last chunk
            
            #10;
            
            // Count actual ones
            expected_count = 0;
            for (j = 0; j < 255; j = j + 1) begin
                if (in[j]) expected_count = expected_count + 1;
            end
            
            expected = expected_count;
            
            $display("Random %2d: count=%3d (expect %3d) %s", 
                i, out, expected,
                (out == expected) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (out == expected) num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 7: Edge cases
        $display("\nTest 7: Edge Cases");
        $display("---------------------------------------------------------------");
        
        // Maximum count minus 1
        in = {255{1'b1}};
        in[100] = 1'b0;  // Clear one bit
        #10;
        expected = 8'd254;
        check_result("254 ones", expected);
        
        // Half set (127 and 128)
        in = 255'h0;
        for (i = 0; i < 127; i = i + 1) begin
            in[i] = 1'b1;
        end
        #10;
        expected = 8'd127;
        check_result("Exactly 127 ones", expected);
        
        in[127] = 1'b1;
        #10;
        expected = 8'd128;
        check_result("Exactly 128 ones", expected);
        
        // Test 8: Stress patterns
        $display("\nTest 8: Stress Pattern Tests");
        $display("---------------------------------------------------------------");
        
        // Checkerboard patterns
        for (k = 1; k <= 8; k = k * 2) begin
            in = 255'h0;
            for (i = 0; i < 255; i = i + 1) begin
                if ((i / k) % 2 == 0) in[i] = 1'b1;
            end
            #10;
            
            // Count actual ones
            actual_ones = 0;
            for (j = 0; j < 255; j = j + 1) begin
                if (in[j]) actual_ones = actual_ones + 1;
            end
            
            $display("Checkerboard (size=%2d): count=%3d %s", 
                k, out,
                (out == actual_ones) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (out == actual_ones) num_tests_passed = num_tests_passed + 1;
        end
        
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
        input [7:0] expected_value;
        begin
            total_tests = total_tests + 1;
            
            if (out == expected_value) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%0s: out=%3d (expect %3d) PASS", 
                    description, out, expected_value);
            end else begin
                $display("%0s: out=%3d (expect %3d) FAIL", 
                    description, out, expected_value);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("popcount255_tb.vcd");
        $dumpvars(0, tb_top_module_popcount255);
    end
    
endmodule