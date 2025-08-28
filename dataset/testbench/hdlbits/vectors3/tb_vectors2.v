`timescale 1ns/1ps

module tb_top_module();
    // Inputs (driven by testbench)
    reg [31:0] in;
    
    // Outputs (driven by DUT)
    wire [31:0] out;
    
    // Instantiate the Design Under Test (DUT)
    top_module dut (
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
        $display("Testing Byte-Order Reversal Module (Endianness Converter)");
        $display("Function: Reverses byte order of 32-bit word");
        $display("in[31:24] -> out[7:0], in[23:16] -> out[15:8]");
        $display("in[15:8] -> out[23:16], in[7:0] -> out[31:24]");
        $display("===============================================================");
        $display("Time |      in        |   Expected     |    Actual      | Result");
        $display("---------------------------------------------------------------");
        
        // Test Case 1: All zeros
        in = 32'h00000000;
        expected_out = 32'h00000000;
        #10;
        check_output("All zeros", expected_out);
        
        // Test Case 2: All ones
        in = 32'hFFFFFFFF;
        expected_out = 32'hFFFFFFFF;
        #10;
        check_output("All ones", expected_out);
        
        // Test Case 3: Sequential bytes
        in = 32'h01234567;
        expected_out = 32'h67452301;
        #10;
        check_output("Sequential bytes", expected_out);
        
        // Test Case 4: Reverse sequential
        in = 32'h89ABCDEF;
        expected_out = 32'hEFCDAB89;
        #10;
        check_output("Reverse sequential", expected_out);
        
        // Test Case 5: Single byte patterns
        $display("\n===============================================================");
        $display("Single Byte Pattern Tests:");
        $display("---------------------------------------------------------------");
        
        // First byte only
        in = 32'h000000FF;
        expected_out = 32'hFF000000;
        #10;
        check_output("First byte only", expected_out);
        
        // Second byte only
        in = 32'h0000FF00;
        expected_out = 32'h00FF0000;
        #10;
        check_output("Second byte only", expected_out);
        
        // Third byte only
        in = 32'h00FF0000;
        expected_out = 32'h0000FF00;
        #10;
        check_output("Third byte only", expected_out);
        
        // Fourth byte only
        in = 32'hFF000000;
        expected_out = 32'h000000FF;
        #10;
        check_output("Fourth byte only", expected_out);
        
        // Test Case 6: Common patterns
        $display("\n===============================================================");
        $display("Common Pattern Tests:");
        $display("---------------------------------------------------------------");
        
        // DEADBEEF pattern
        in = 32'hDEADBEEF;
        expected_out = 32'hEFBEADDE;
        #10;
        check_output("0xDEADBEEF", expected_out);
        
        // CAFEBABE pattern
        in = 32'hCAFEBABE;
        expected_out = 32'hBEBAFECA;
        #10;
        check_output("0xCAFEBABE", expected_out);
        
        // Test Case 7: Palindrome check
        in = 32'h12211221;
        expected_out = 32'h21122112;
        #10;
        check_output("Palindrome bytes", expected_out);
        
        // Test Case 8: Alternating patterns
        in = 32'hAAAAAAAA;
        expected_out = 32'hAAAAAAAA;
        #10;
        check_output("All 0xAA", expected_out);
        
        in = 32'h55555555;
        expected_out = 32'h55555555;
        #10;
        check_output("All 0x55", expected_out);
        
        // Test Case 9: Walking bytes
        $display("\n===============================================================");
        $display("Walking Byte Test:");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 4; i = i + 1) begin
            in = 32'hFF << (i * 8);
            case(i)
                0: expected_out = 32'hFF000000;
                1: expected_out = 32'h00FF0000;
                2: expected_out = 32'h0000FF00;
                3: expected_out = 32'h000000FF;
            endcase
            #10;
            $display("Byte %0d: in=%08h => out=%08h", i, in, out);
            if (out == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
        end
        
        // Test Case 10: Double reversal (should return to original)
        $display("\n===============================================================");
        $display("Double Reversal Test (Verify Reversibility):");
        $display("---------------------------------------------------------------");
        
        in = 32'h12345678;
        #10;
        $display("Original: %08h", in);
        $display("Reversed: %08h", out);
        in = out;  // Feed output back as input
        #10;
        $display("Double reversed: %08h (should be 12345678)", out);
        if (out == 32'h12345678) begin
            $display("Double reversal: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Double reversal: FAIL");
        end
        total_tests = total_tests + 1;
        
        // Test Case 11: Network byte order examples
        $display("\n===============================================================");
        $display("Network/Host Byte Order Examples:");
        $display("---------------------------------------------------------------");
        
        // IP address example (192.168.1.1)
        in = 32'hC0A80101;  // 192.168.1.1 in hex
        expected_out = 32'h0101A8C0;
        #10;
        check_output("IP: 192.168.1.1", expected_out);
        
        // Port number examples
        in = 32'h00000050;  // Port 80 (HTTP)
        expected_out = 32'h50000000;
        #10;
        check_output("Port 80 (0x50)", expected_out);
        
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
    
    // Task to check output and display results
    task check_output;
        input [30*8:1] test_name;
        input [31:0] exp_out;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  | %08h | %08h | %08h | %s", 
                $time, in, exp_out, out,
                (out == exp_out) ? "PASS" : "FAIL");
            
            if (out == exp_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("byte_reversal_tb.vcd");
        $dumpvars(0, tb_top_module);
    end
    
endmodule