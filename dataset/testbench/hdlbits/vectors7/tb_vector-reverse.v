`timescale 1ns/1ps

module tb_vector_reverse();
    // Inputs (driven by testbench)
    reg [7:0] in;
    
    // Outputs (driven by DUT)
    wire [7:0] out;
    
    // Instantiate the Design Under Test (DUT)
    vector_reverse dut (
        .in(in),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected value
    reg [7:0] expected_out;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Vector Reverse Module");
        $display("Function: Reverses bit order of 8-bit input");
        $display("in[7:0] => out[0:7] (MSB becomes LSB, etc.)");
        $display("===============================================================");
        $display("Time |   in[7:0]  | Expected out | Actual out | Result");
        $display("-------------------------------------------------------");
        
        // Test Case 1: All zeros
        in = 8'b00000000;
        expected_out = 8'b00000000;
        #10;
        check_result("All zeros", expected_out);
        
        // Test Case 2: All ones
        in = 8'b11111111;
        expected_out = 8'b11111111;
        #10;
        check_result("All ones", expected_out);
        
        // Test Case 3: Alternating pattern 10101010
        in = 8'b10101010;
        expected_out = 8'b01010101;
        #10;
        check_result("Alternating 10", expected_out);
        
        // Test Case 4: Alternating pattern 01010101
        in = 8'b01010101;
        expected_out = 8'b10101010;
        #10;
        check_result("Alternating 01", expected_out);
        
        // Test Case 5: Single bit tests (walking 1)
        $display("\n===============================================================");
        $display("Walking 1 Test:");
        $display("---------------------------------------------------------------");
        for (i = 0; i < 8; i = i + 1) begin
            in = 8'h01 << i;
            expected_out = 8'h80 >> i;
            #10;
            $display("Bit %0d: in=%08b => out=%08b (bit %0d to bit %0d)",
                i, in, out, i, 7-i);
            if (out == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
        end
        
        // Test Case 6: Walking 0 (all 1s except one 0)
        $display("\n===============================================================");
        $display("Walking 0 Test:");
        $display("---------------------------------------------------------------");
        for (i = 0; i < 8; i = i + 1) begin
            in = ~(8'h01 << i);
            expected_out = ~(8'h80 >> i);
            #10;
            $display("Bit %0d: in=%08b => out=%08b",
                i, in, out);
            if (out == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
        end
        
        // Test Case 7: Palindrome patterns (should equal themselves when reversed)
        $display("\n===============================================================");
        $display("Palindrome Test (output should equal input):");
        $display("---------------------------------------------------------------");
        
        in = 8'b10000001;  // Palindrome
        #10;
        $display("in=%08b, out=%08b %s", in, out, (in == out) ? "✓ Palindrome" : "✗");
        
        in = 8'b11000011;  // Palindrome
        #10;
        $display("in=%08b, out=%08b %s", in, out, (in == out) ? "✓ Palindrome" : "✗");
        
        in = 8'b10111101;  // Palindrome
        #10;
        $display("in=%08b, out=%08b %s", in, out, (in == out) ? "✓ Palindrome" : "✗");
        
        in = 8'b01100110;  // Palindrome
        #10;
        $display("in=%08b, out=%08b %s", in, out, (in == out) ? "✓ Palindrome" : "✗");
        
        // Test Case 8: Common byte values
        $display("\n===============================================================");
        $display("Common Byte Values Test:");
        $display("---------------------------------------------------------------");
        
        // Test 0x0F (00001111)
        in = 8'h0F;
        expected_out = 8'hF0;
        #10;
        check_result("0x0F", expected_out);
        
        // Test 0xF0 (11110000)
        in = 8'hF0;
        expected_out = 8'h0F;
        #10;
        check_result("0xF0", expected_out);
        
        // Test 0x55 (01010101)
        in = 8'h55;
        expected_out = 8'hAA;
        #10;
        check_result("0x55", expected_out);
        
        // Test 0xAA (10101010)
        in = 8'hAA;
        expected_out = 8'h55;
        #10;
        check_result("0xAA", expected_out);
        
        // Test Case 9: ASCII character reversals
        $display("\n===============================================================");
        $display("ASCII Character Bit Reversal:");
        $display("---------------------------------------------------------------");
        
        in = 8'h41;  // 'A' = 01000001
        expected_out = 8'h82;  // 10000010
        #10;
        $display("'A' (0x41): in=%08b => out=%08b (0x%02X)", in, out, out);
        
        in = 8'h30;  // '0' = 00110000
        expected_out = 8'h0C;  // 00001100
        #10;
        $display("'0' (0x30): in=%08b => out=%08b (0x%02X)", in, out, out);
        
        // Test Case 10: Double reversal test
        $display("\n===============================================================");
        $display("Double Reversal Test (should return to original):");
        $display("---------------------------------------------------------------");
        
        in = 8'b11010010;
        #10;
        $display("Original: %08b", in);
        $display("Reversed: %08b", out);
        in = out;  // Feed output back as input
        #10;
        $display("Double reversed: %08b %s", out, 
            (out == 8'b11010010) ? "✓ Matches original" : "✗ Doesn't match");
        
        // Test Case 11: Bit position mapping verification
        $display("\n===============================================================");
        $display("Bit Position Mapping:");
        $display("---------------------------------------------------------------");
        in = 8'b10110011;
        #10;
        $display("Input:  in[7]=%b in[6]=%b in[5]=%b in[4]=%b in[3]=%b in[2]=%b in[1]=%b in[0]=%b",
            in[7], in[6], in[5], in[4], in[3], in[2], in[1], in[0]);
        $display("Output: out[7]=%b out[6]=%b out[5]=%b out[4]=%b out[3]=%b out[2]=%b out[1]=%b out[0]=%b",
            out[7], out[6], out[5], out[4], out[3], out[2], out[1], out[0]);
        $display("Mapping: in[0]→out[7], in[1]→out[6], ..., in[7]→out[0]");
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else if (num_tests_passed != 0)
            $display("Overall Result: SOME TESTS PASSED ⚠");
        else
            $display("Overall Result: NO TESTS PASSED ✗");
        $display("===============================================================");
        
        $finish;
    end
    
    // Task to check result and update counters
    task check_result;
        input [20*8:1] test_name;
        input [7:0] exp_out;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  | %08b |   %08b  |  %08b  | %s",
                $time, in, exp_out, out,
                (out == exp_out) ? "PASS" : "FAIL");
            
            if (out == exp_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("vector_reverse_tb.vcd");
        $dumpvars(0, tb_vector_reverse);
    end
    
endmodule