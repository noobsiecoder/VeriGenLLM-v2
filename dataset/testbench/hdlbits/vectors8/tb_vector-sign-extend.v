`timescale 1ns/1ps

module tb_sign_extend_8to32();
    // Inputs (driven by testbench)
    reg [7:0] in;
    
    // Outputs (driven by DUT)
    wire [31:0] out;
    
    // Instantiate the Design Under Test (DUT)
    sign_extend_8to32 dut (
        .in(in),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected value
    reg [31:0] expected_out;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Sign Extension Module (8-bit to 32-bit)");
        $display("Function: Extends 8-bit signed value to 32-bit signed value");
        $display("===============================================================");
        $display("Time |  in (hex) | in (dec) | Expected out (hex) | Actual out (hex) | Result");
        $display("-----------------------------------------------------------------------------");
        
        // Test Case 1: Zero
        in = 8'h00;
        expected_out = 32'h00000000;
        #10;
        check_result("Zero", expected_out);
        
        // Test Case 2: Positive maximum (127)
        in = 8'h7F;  // 01111111
        expected_out = 32'h0000007F;
        #10;
        check_result("Max positive (127)", expected_out);
        
        // Test Case 3: Negative one (-1)
        in = 8'hFF;  // 11111111
        expected_out = 32'hFFFFFFFF;
        #10;
        check_result("Negative one (-1)", expected_out);
        
        // Test Case 4: Most negative (-128)
        in = 8'h80;  // 10000000
        expected_out = 32'hFFFFFF80;
        #10;
        check_result("Most negative (-128)", expected_out);
        
        // Test Case 5: Small positive values
        $display("\n===============================================================");
        $display("Small Positive Values (sign bit = 0):");
        $display("---------------------------------------------------------------");
        
        for (i = 1; i <= 5; i = i + 1) begin
            in = i[7:0];
            expected_out = {24'h000000, in};
            #10;
            $display("%3t  |    %02h     |   %3d    |     %08h       |    %08h     | %s",
                $time, in, $signed(in), expected_out, out,
                (out == expected_out) ? "PASS" : "FAIL");
            if (out == expected_out) num_tests_passed = num_tests_passed + 1;
            total_tests = total_tests + 1;
        end
        
        // Test Case 6: Small negative values
        $display("\n===============================================================");
        $display("Small Negative Values (sign bit = 1):");
        $display("---------------------------------------------------------------");
        
        for (i = -1; i >= -5; i = i - 1) begin
            in = i[7:0];
            expected_out = {24'hFFFFFF, in};
            #10;
            $display("%3t  |    %02h     |   %3d    |     %08h       |    %08h     | %s",
                $time, in, $signed(in), expected_out, out,
                (out == expected_out) ? "PASS" : "FAIL");
            if (out == expected_out) num_tests_passed = num_tests_passed + 1;
            total_tests = total_tests + 1;
        end
        
        // Test Case 7: Boundary values around sign change
        $display("\n===============================================================");
        $display("Boundary Values Around Sign Change:");
        $display("---------------------------------------------------------------");
        
        // Just below sign change (positive)
        in = 8'h7E;  // 126
        expected_out = 32'h0000007E;
        #10;
        check_result_with_decimal("126", expected_out);
        
        in = 8'h7F;  // 127
        expected_out = 32'h0000007F;
        #10;
        check_result_with_decimal("127", expected_out);
        
        // Just above sign change (negative)
        in = 8'h80;  // -128
        expected_out = 32'hFFFFFF80;
        #10;
        check_result_with_decimal("-128", expected_out);
        
        in = 8'h81;  // -127
        expected_out = 32'hFFFFFF81;
        #10;
        check_result_with_decimal("-127", expected_out);
        
        // Test Case 8: Pattern verification
        $display("\n===============================================================");
        $display("Sign Extension Pattern Verification:");
        $display("---------------------------------------------------------------");
        
        // Positive number (sign bit = 0)
        in = 8'b01010101;  // 0x55
        #10;
        $display("Positive: in=%08b", in);
        $display("         out=%032b", out);
        $display("         Upper 24 bits: %024b (all zeros)", out[31:8]);
        
        // Negative number (sign bit = 1)
        in = 8'b10101010;  // 0xAA
        #10;
        $display("\nNegative: in=%08b", in);
        $display("         out=%032b", out);
        $display("         Upper 24 bits: %024b (all ones)", out[31:8]);
        
        // Test Case 9: Arithmetic verification
        $display("\n===============================================================");
        $display("Arithmetic Verification (8-bit vs 32-bit signed):");
        $display("---------------------------------------------------------------");
        
        in = 8'd50;
        #10;
        $display("Positive: 8-bit: %d, 32-bit: %d", $signed(in), $signed(out));
        
        in = -8'd50;
        #10;
        $display("Negative: 8-bit: %d, 32-bit: %d", $signed(in), $signed(out));
        
        // Test Case 10: All possible sign bit = 0 values
        $display("\n===============================================================");
        $display("Testing MSB propagation:");
        $display("---------------------------------------------------------------");
        
        in = 8'b00000000;
        #10;
        $display("MSB=0: Upper 24 bits = %06h (should be 000000)", out[31:8]);
        
        in = 8'b10000000;
        #10;
        $display("MSB=1: Upper 24 bits = %06h (should be FFFFFF)", out[31:8]);
        
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
        input [30*8:1] test_name;
        input [31:0] exp_out;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  |    %02h     |   %3d    |     %08h       |    %08h     | %s",
                $time, in, $signed(in), exp_out, out,
                (out == exp_out) ? "PASS" : "FAIL");
            
            if (out == exp_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Task with decimal display
    task check_result_with_decimal;
        input [10*8:1] decimal_str;
        input [31:0] exp_out;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  |    %02h     |  %4s   |     %08h       |    %08h     | %s",
                $time, in, decimal_str, exp_out, out,
                (out == exp_out) ? "PASS" : "FAIL");
            
            if (out == exp_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("sign_extend_tb.vcd");
        $dumpvars(0, tb_sign_extend_8to32);
    end
    
endmodule