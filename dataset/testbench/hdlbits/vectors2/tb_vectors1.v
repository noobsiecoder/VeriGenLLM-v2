`timescale 1ns/1ps

module tb_top_module();
    // Inputs (driven by testbench)
    reg [15:0] in;
    
    // Outputs (driven by DUT)
    wire [7:0] out_hi;
    wire [7:0] out_lo;
    
    // Instantiate the Design Under Test (DUT)
    top_module dut (
        .in(in),
        .out_hi(out_hi),
        .out_lo(out_lo)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected values
    reg [7:0] expected_hi;
    reg [7:0] expected_lo;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Half-Word Splitter Module");
        $display("Function: out_hi = in[15:8], out_lo = in[7:0]");
        $display("===============================================================");
        $display("Time |    in[15:0]    | Expected Hi/Lo | Actual Hi/Lo | Result");
        $display("---------------------------------------------------------------");
        
        // Test Case 1: All zeros
        in = 16'h0000;
        expected_hi = 8'h00;
        expected_lo = 8'h00;
        #10;
        check_outputs("All zeros", expected_hi, expected_lo);
        
        // Test Case 2: All ones
        in = 16'hFFFF;
        expected_hi = 8'hFF;
        expected_lo = 8'hFF;
        #10;
        check_outputs("All ones", expected_hi, expected_lo);
        
        // Test Case 3: Upper byte only
        in = 16'hFF00;
        expected_hi = 8'hFF;
        expected_lo = 8'h00;
        #10;
        check_outputs("Upper byte only", expected_hi, expected_lo);
        
        // Test Case 4: Lower byte only
        in = 16'h00FF;
        expected_hi = 8'h00;
        expected_lo = 8'hFF;
        #10;
        check_outputs("Lower byte only", expected_hi, expected_lo);
        
        // Test Case 5: Alternating bits pattern
        in = 16'hAAAA;
        expected_hi = 8'hAA;
        expected_lo = 8'hAA;
        #10;
        check_outputs("Alternating 1010", expected_hi, expected_lo);
        
        // Test Case 6: Alternating bits pattern (inverted)
        in = 16'h5555;
        expected_hi = 8'h55;
        expected_lo = 8'h55;
        #10;
        check_outputs("Alternating 0101", expected_hi, expected_lo);
        
        // Test Case 7: Different bytes
        in = 16'hDEAD;
        expected_hi = 8'hDE;
        expected_lo = 8'hAD;
        #10;
        check_outputs("0xDEAD pattern", expected_hi, expected_lo);
        
        // Test Case 8: Another different pattern
        in = 16'hBEEF;
        expected_hi = 8'hBE;
        expected_lo = 8'hEF;
        #10;
        check_outputs("0xBEEF pattern", expected_hi, expected_lo);
        
        // Test Case 9: Incrementing pattern
        in = 16'h0123;
        expected_hi = 8'h01;
        expected_lo = 8'h23;
        #10;
        check_outputs("0x0123 pattern", expected_hi, expected_lo);
        
        // Test Case 10: Walking ones test
        $display("\n===============================================================");
        $display("Walking Ones Test:");
        $display("---------------------------------------------------------------");
        for (i = 0; i < 16; i = i + 1) begin
            in = 16'h0001 << i;
            expected_hi = in[15:8];
            expected_lo = in[7:0];
            #10;
            $display("Bit %2d: in=%04h => hi=%02h, lo=%02h", 
                i, in, out_hi, out_lo);
            if (out_hi == expected_hi && out_lo == expected_lo) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
        end
        
        // Test Case 11: Random patterns
        $display("\n===============================================================");
        $display("Random Pattern Tests:");
        $display("---------------------------------------------------------------");
        
        // Random pattern 1
        in = 16'h1234;
        expected_hi = 8'h12;
        expected_lo = 8'h34;
        #10;
        check_outputs("0x1234", expected_hi, expected_lo);
        
        // Random pattern 2
        in = 16'h5678;
        expected_hi = 8'h56;
        expected_lo = 8'h78;
        #10;
        check_outputs("0x5678", expected_hi, expected_lo);
        
        // Random pattern 3
        in = 16'h9ABC;
        expected_hi = 8'h9A;
        expected_lo = 8'hBC;
        #10;
        check_outputs("0x9ABC", expected_hi, expected_lo);
        
        // Random pattern 4
        in = 16'hF0E1;
        expected_hi = 8'hF0;
        expected_lo = 8'hE1;
        #10;
        check_outputs("0xF0E1", expected_hi, expected_lo);
        
        // Test Case 12: Boundary values
        $display("\n===============================================================");
        $display("Boundary Value Tests:");
        $display("---------------------------------------------------------------");
        
        // Min value
        in = 16'h0000;
        #10;
        $display("Min: in=%04h => hi=%02h, lo=%02h", in, out_hi, out_lo);
        
        // Max value
        in = 16'hFFFF;
        #10;
        $display("Max: in=%04h => hi=%02h, lo=%02h", in, out_hi, out_lo);
        
        // Middle value
        in = 16'h8000;
        #10;
        $display("Mid: in=%04h => hi=%02h, lo=%02h", in, out_hi, out_lo);
        
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
    
    // Task to check outputs and display results
    task check_outputs;
        input [20*8:1] test_name;
        input [7:0] exp_hi;
        input [7:0] exp_lo;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  |     %04h       |    %02h/%02h    |   %02h/%02h   | %s", 
                $time, in, exp_hi, exp_lo, out_hi, out_lo,
                (out_hi == exp_hi && out_lo == exp_lo) ? "PASS" : "FAIL");
            
            if (out_hi == exp_hi && out_lo == exp_lo) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("halfword_splitter_tb.vcd");
        $dumpvars(0, tb_top_module);
    end
    
endmodule