`timescale 1ns/1ps

module tb_vector_gates();
    // Inputs (driven by testbench)
    reg [2:0] a;
    reg [2:0] b;
    
    // Outputs (driven by DUT)
    wire [2:0] out_or_bitwise;
    wire out_or_logical;
    wire [5:0] out_not;
    
    // Instantiate the Design Under Test (DUT)
    vector_gates dut (
        .a(a),
        .b(b),
        .out_or_bitwise(out_or_bitwise),
        .out_or_logical(out_or_logical),
        .out_not(out_not)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    
    // Expected values
    reg [2:0] expected_or_bitwise;
    reg expected_or_logical;
    reg [5:0] expected_not;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Vector Gates Module");
        $display("Functions:");
        $display("  - Bitwise OR: out_or_bitwise = a | b");
        $display("  - Logical OR: out_or_logical = a || b");
        $display("  - NOT gates: out_not[2:0] = ~a, out_not[5:3] = ~b");
        $display("===============================================================");
        $display("Time | a[2:0] | b[2:0] | Bitwise OR | Logical OR | NOT(a,b) | Result");
        $display("---------------------------------------------------------------------");
        
        // Test all 64 combinations (8 values for a × 8 values for b)
        for (i = 0; i < 8; i = i + 1) begin
            for (j = 0; j < 8; j = j + 1) begin
                a = i[2:0];
                b = j[2:0];
                
                // Calculate expected values
                expected_or_bitwise = a | b;
                expected_or_logical = (a != 0) || (b != 0);
                expected_not[2:0] = ~a;
                expected_not[5:3] = ~b;
                
                // Wait for propagation
                #10;
                
                // Check all outputs
                if (out_or_bitwise == expected_or_bitwise && 
                    out_or_logical == expected_or_logical && 
                    out_not == expected_not) begin
                    num_tests_passed = num_tests_passed + 1;
                end
                total_tests = total_tests + 1;
                
                // Display result for each test
                $display("%3t  |  %b  |  %b  |     %b     |     %b      | %b | %s",
                    $time, a, b, out_or_bitwise, out_or_logical, out_not,
                    (out_or_bitwise == expected_or_bitwise && 
                     out_or_logical == expected_or_logical && 
                     out_not == expected_not) ? "PASS" : "FAIL");
            end
        end
        
        // Special test cases with detailed output
        $display("\n===============================================================");
        $display("Special Test Cases:");
        $display("---------------------------------------------------------------");
        
        // Test Case 1: Both zero (logical OR should be 0)
        a = 3'b000;
        b = 3'b000;
        #10;
        $display("\nBoth zeros:");
        $display("  Inputs: a=%b, b=%b", a, b);
        $display("  Bitwise OR: %b (expected: 000)", out_or_bitwise);
        $display("  Logical OR: %b (expected: 0)", out_or_logical);
        $display("  NOT outputs: %b (expected: 111_111)", out_not);
        
        // Test Case 2: One zero, one non-zero (logical OR should be 1)
        a = 3'b000;
        b = 3'b101;
        #10;
        $display("\nOne zero, one non-zero:");
        $display("  Inputs: a=%b, b=%b", a, b);
        $display("  Bitwise OR: %b (expected: 101)", out_or_bitwise);
        $display("  Logical OR: %b (expected: 1)", out_or_logical);
        $display("  NOT outputs: %b (expected: 010_111)", out_not);
        
        // Test Case 3: Both non-zero
        a = 3'b110;
        b = 3'b011;
        #10;
        $display("\nBoth non-zero:");
        $display("  Inputs: a=%b, b=%b", a, b);
        $display("  Bitwise OR: %b (expected: 111)", out_or_bitwise);
        $display("  Logical OR: %b (expected: 1)", out_or_logical);
        $display("  NOT outputs: %b (expected: 100_001)", out_not);
        
        // Test Case 4: Demonstrating bitwise vs logical difference
        $display("\n===============================================================");
        $display("Bitwise vs Logical OR Comparison:");
        $display("---------------------------------------------------------------");
        $display("a    | b    | a|b  | a||b | Comment");
        $display("-----|------|------|------|--------");
        
        a = 3'b010; b = 3'b100; #10;
        $display("%b | %b | %b  | %b    | Different bits set", a, b, out_or_bitwise, out_or_logical);
        
        a = 3'b111; b = 3'b111; #10;
        $display("%b | %b | %b  | %b    | All bits set", a, b, out_or_bitwise, out_or_logical);
        
        a = 3'b001; b = 3'b000; #10;
        $display("%b | %b | %b  | %b    | Only LSB of a set", a, b, out_or_bitwise, out_or_logical);
        
        // Test Case 5: NOT operation verification
        $display("\n===============================================================");
        $display("NOT Operation Verification:");
        $display("---------------------------------------------------------------");
        $display("a    | ~a   | b    | ~b   | Combined NOT output");
        $display("-----|------|------|------|--------------------");
        
        for (i = 0; i < 8; i = i + 1) begin
            a = i[2:0];
            b = ~i[2:0];  // b is complement of a
            #10;
            $display("%b | %b  | %b | %b  | %b", 
                a, out_not[2:0], b, out_not[5:3], out_not);
        end
        
        // Test Case 6: Walking ones pattern
        $display("\n===============================================================");
        $display("Walking Ones Pattern Test:");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 3; i = i + 1) begin
            a = 3'b001 << i;
            b = 3'b000;
            #10;
            $display("a=%b, b=%b => OR bitwise=%b, OR logical=%b, NOT=%b",
                a, b, out_or_bitwise, out_or_logical, out_not);
        end
        
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
        $dumpfile("vector_gates_tb.vcd");
        $dumpvars(0, tb_vector_gates);
    end
    
endmodule