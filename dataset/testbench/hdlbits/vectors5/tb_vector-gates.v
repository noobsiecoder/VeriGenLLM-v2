`timescale 1ns/1ps

module tb_four_bit_vector_gates();
    // Inputs (driven by testbench)
    reg [3:0] in;
    
    // Outputs (driven by DUT)
    wire out_and;
    wire out_or;
    wire out_xor;
    
    // Instantiate the Design Under Test (DUT)
    four_bit_vector_gates dut (
        .in(in),
        .out_and(out_and),
        .out_or(out_or),
        .out_xor(out_xor)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    integer ones_count;
    
    // Expected values
    reg expected_and;
    reg expected_or;
    reg expected_xor;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 4-bit Vector Reduction Gates Module");
        $display("Functions:");
        $display("  - AND reduction: out_and = &in (all bits must be 1)");
        $display("  - OR reduction:  out_or = |in (at least one bit is 1)");
        $display("  - XOR reduction: out_xor = ^in (odd number of 1s)");
        $display("===============================================================");
        $display("Time | in[3:0] | AND | OR | XOR | Expected AND/OR/XOR | Result");
        $display("---------------------------------------------------------------");
        
        // Test all 16 possible input combinations
        for (i = 0; i < 16; i = i + 1) begin
            in = i[3:0];
            
            // Calculate expected values
            expected_and = (in == 4'b1111) ? 1'b1 : 1'b0;
            expected_or = (in != 4'b0000) ? 1'b1 : 1'b0;
            
            // Count ones for XOR (odd parity)
            ones_count = in[0] + in[1] + in[2] + in[3];
            expected_xor = ones_count[0]; // LSB gives odd/even
            
            // Wait for propagation
            #10;
            
            // Check all outputs
            if (out_and == expected_and && out_or == expected_or && out_xor == expected_xor) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%3t  |  %b  |  %b  | %b  |  %b  |        %b/%b/%b       | PASS",
                    $time, in, out_and, out_or, out_xor, 
                    expected_and, expected_or, expected_xor);
            end else begin
                $display("%3t  |  %b  |  %b  | %b  |  %b  |        %b/%b/%b       | FAIL",
                    $time, in, out_and, out_or, out_xor, 
                    expected_and, expected_or, expected_xor);
            end
            total_tests = total_tests + 1;
        end
        
        // Special test cases with detailed analysis
        $display("\n===============================================================");
        $display("Detailed Analysis of Key Cases:");
        $display("---------------------------------------------------------------");
        
        // All zeros
        in = 4'b0000;
        #10;
        $display("\nAll zeros (0000):");
        $display("  AND: %b (0 - need all bits to be 1)", out_and);
        $display("  OR:  %b (0 - no bits are 1)", out_or);
        $display("  XOR: %b (0 - even number (0) of 1s)", out_xor);
        
        // All ones
        in = 4'b1111;
        #10;
        $display("\nAll ones (1111):");
        $display("  AND: %b (1 - all bits are 1)", out_and);
        $display("  OR:  %b (1 - at least one bit is 1)", out_or);
        $display("  XOR: %b (0 - even number (4) of 1s)", out_xor);
        
        // Single one
        in = 4'b0001;
        #10;
        $display("\nSingle one (0001):");
        $display("  AND: %b (0 - not all bits are 1)", out_and);
        $display("  OR:  %b (1 - at least one bit is 1)", out_or);
        $display("  XOR: %b (1 - odd number (1) of 1s)", out_xor);
        
        // Two ones
        in = 4'b0011;
        #10;
        $display("\nTwo ones (0011):");
        $display("  AND: %b (0 - not all bits are 1)", out_and);
        $display("  OR:  %b (1 - at least one bit is 1)", out_or);
        $display("  XOR: %b (0 - even number (2) of 1s)", out_xor);
        
        // Three ones
        in = 4'b0111;
        #10;
        $display("\nThree ones (0111):");
        $display("  AND: %b (0 - not all bits are 1)", out_and);
        $display("  OR:  %b (1 - at least one bit is 1)", out_or);
        $display("  XOR: %b (1 - odd number (3) of 1s)", out_xor);
        
        // Walking ones test
        $display("\n===============================================================");
        $display("Walking Ones Test:");
        $display("---------------------------------------------------------------");
        $display("Position | in[3:0] | AND | OR | XOR | Comment");
        $display("---------|---------|-----|----|----|--------");
        
        for (i = 0; i < 4; i = i + 1) begin
            in = 4'b0001 << i;
            #10;
            $display("   %0d     |  %b  |  %b  | %b  |  %b  | Single 1 at bit %0d",
                i, in, out_and, out_or, out_xor, i);
        end
        
        // XOR parity demonstration
        $display("\n===============================================================");
        $display("XOR Parity Demonstration:");
        $display("---------------------------------------------------------------");
        $display("in[3:0] | # of 1s | XOR | Parity");
        $display("--------|---------|-----|-------");
        
        in = 4'b0000; #10;
        $display(" %b   |    0    |  %b  | Even", in, out_xor);
        
        in = 4'b1000; #10;
        $display(" %b   |    1    |  %b  | Odd", in, out_xor);
        
        in = 4'b1100; #10;
        $display(" %b   |    2    |  %b  | Even", in, out_xor);
        
        in = 4'b1110; #10;
        $display(" %b   |    3    |  %b  | Odd", in, out_xor);
        
        in = 4'b1111; #10;
        $display(" %b   |    4    |  %b  | Even", in, out_xor);
        
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
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("four_bit_vector_gates_tb.vcd");
        $dumpvars(0, tb_four_bit_vector_gates);
    end
    
endmodule