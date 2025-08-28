`timescale 1ns/1ps

module tb_xnor_matrix();
    // Inputs (driven by testbench)
    reg a, b, c, d, e;
    
    // Outputs (driven by DUT)
    wire [24:0] out;
    
    // Instantiate the Design Under Test (DUT)
    xnor_matrix dut (
        .a(a), .b(b), .c(c), .d(d), .e(e),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected values
    reg [24:0] expected_out;
    reg [24:0] replicated_inputs;
    reg [24:0] repeated_pattern;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing XNOR Matrix Module");
        $display("Function: out = ~{5{a}, 5{b}, 5{c}, 5{d}, 5{e}} ^ {5{a,b,c,d,e}}");
        $display("===============================================================");
        
        // Explain the operation
        $display("\nOperation breakdown:");
        $display("1. First operand: ~{5{a}, 5{b}, 5{c}, 5{d}, 5{e}}");
        $display("   - Replicates each input 5 times, then inverts");
        $display("2. Second operand: {5{a,b,c,d,e}}");
        $display("   - Replicates the 5-bit pattern 5 times");
        $display("3. Result: XNOR of the two operands\n");
        
        $display("Time | a b c d e | First Operand          | Second Operand         | Output (XNOR)          | Result");
        $display("-----------------------------------------------------------------------------------------------------");
        
        // Test all 32 possible input combinations
        for (i = 0; i < 32; i = i + 1) begin
            {a, b, c, d, e} = i[4:0];
            
            // Calculate expected values
            replicated_inputs = { {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} };
            repeated_pattern = { {5{a,b,c,d,e}} };
            expected_out = ~replicated_inputs ^ repeated_pattern;
            
            #10;
            
            if (out == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
            
            // Display for interesting cases
            if (i < 5 || i == 15 || i == 16 || i == 31) begin
                $display("%3t  | %b %b %b %b %b | %025b | %025b | %025b | %s",
                    $time, a, b, c, d, e, ~replicated_inputs, repeated_pattern, out,
                    (out == expected_out) ? "PASS" : "FAIL");
            end
        end
        
        // Detailed analysis of specific patterns
        $display("\n===============================================================");
        $display("Detailed Analysis of Specific Patterns:");
        $display("---------------------------------------------------------------");
        
        // Test Case 1: All zeros
        {a, b, c, d, e} = 5'b00000;
        #10;
        $display("\nAll zeros (00000):");
        $display("  Replicated inputs: %025b", { {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Inverted:          %025b", ~{ {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Repeated pattern:  %025b", { {5{a,b,c,d,e}} });
        $display("  XNOR result:       %025b", out);
        $display("  Expected all 1s since ~0 ^ 0 = 1");
        
        // Test Case 2: All ones
        {a, b, c, d, e} = 5'b11111;
        #10;
        $display("\nAll ones (11111):");
        $display("  Replicated inputs: %025b", { {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Inverted:          %025b", ~{ {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Repeated pattern:  %025b", { {5{a,b,c,d,e}} });
        $display("  XNOR result:       %025b", out);
        $display("  Expected all 1s since ~1 ^ 1 = 1");
        
        // Test Case 3: Single bit set
        {a, b, c, d, e} = 5'b10000;
        #10;
        $display("\nSingle bit set (10000):");
        $display("  Input pattern: a=%b, b=%b, c=%b, d=%b, e=%b", a, b, c, d, e);
        $display("  Replicated inputs: %025b", { {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Inverted:          %025b", ~{ {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Repeated pattern:  %025b", { {5{a,b,c,d,e}} });
        $display("  XNOR result:       %025b", out);
        
        // Test Case 4: Alternating pattern
        {a, b, c, d, e} = 5'b10101;
        #10;
        $display("\nAlternating pattern (10101):");
        $display("  Replicated inputs: %025b", { {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Inverted:          %025b", ~{ {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Repeated pattern:  %025b", { {5{a,b,c,d,e}} });
        $display("  XNOR result:       %025b", out);
        
        // Bit position analysis
        $display("\n===============================================================");
        $display("Bit Position Analysis:");
        $display("---------------------------------------------------------------");
        {a, b, c, d, e} = 5'b10110;
        #10;
        $display("Input: %b%b%b%b%b", a, b, c, d, e);
        $display("\nFirst operand construction (inverted replication):");
        $display("  Bits [24:20]: ~{5{a}} = ~{5{%b}} = %05b", a, out[24:20]);
        $display("  Bits [19:15]: ~{5{b}} = ~{5{%b}} = %05b", b, out[19:15]);
        $display("  Bits [14:10]: ~{5{c}} = ~{5{%b}} = %05b", c, out[14:10]);
        $display("  Bits [9:5]:   ~{5{d}} = ~{5{%b}} = %05b", d, out[9:5]);
        $display("  Bits [4:0]:   ~{5{e}} = ~{5{%b}} = %05b", e, out[4:0]);
        
        $display("\nSecond operand (pattern repetition):");
        $display("  Pattern %b%b%b%b%b repeated 5 times", a, b, c, d, e);
        
        // XNOR truth table reminder
        $display("\n===============================================================");
        $display("XNOR Truth Table Reminder:");
        $display("---------------------------------------------------------------");
        $display("A | B | ~A | ~A XOR B | ~A XNOR B");
        $display("--|---|-------|---------|----------");
        $display("0 | 0 |   1   |    1    |    1");
        $display("0 | 1 |   1   |    0    |    0");
        $display("1 | 0 |   0   |    0    |    0");
        $display("1 | 1 |   0   |    1    |    1");
        $display("\nNote: XNOR(A,B) = ~(A XOR B) = (~A XOR B)");
        
        // Pattern matching test
        $display("\n===============================================================");
        $display("Pattern Matching Test:");
        $display("---------------------------------------------------------------");
        
        // When both operands match (should give all 1s)
        {a, b, c, d, e} = 5'b00000;
        #10;
        $display("When operands match after inversion:");
        $display("  First:  %025b", ~{ {5{a}}, {5{b}}, {5{c}}, {5{d}}, {5{e}} });
        $display("  Second: %025b", { {5{a,b,c,d,e}} });
        $display("  Result: %025b (should have 1s where bits match)", out);
        
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
        $dumpfile("xnor_matrix_tb.vcd");
        $dumpvars(0, tb_xnor_matrix);
    end
    
endmodule