`timescale 1ns/1ps

module tb_wire_decl_gate();
    // Inputs (driven by testbench)
    reg a, b, c, d;
    
    // Outputs (driven by DUT)
    wire out, out_n;
    
    // Instantiate the Design Under Test (DUT)
    wire_decl_gate dut (
        .a(a),
        .b(b),
        .c(c),
        .d(d),
        .out(out),
        .out_n(out_n)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected values
    reg expected_out;
    reg expected_out_n;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing wire_decl_gate module");
        $display("Function: out = (a & b) | (c & d), out_n = ~out");
        $display("===============================================================");
        $display("Time | a | b | c | d | Expected out/out_n | Actual out/out_n | Result");
        $display("---------------------------------------------------------------");
        
        // Test all 16 combinations
        for (i = 0; i < 16; i = i + 1) begin
            // Set inputs based on loop counter
            {a, b, c, d} = i[3:0];
            
            // Calculate expected values
            expected_out = (a & b) | (c & d);
            expected_out_n = ~expected_out;
            
            // Wait for propagation
            #10;
            
            // Check both outputs
            if (out == expected_out && out_n == expected_out_n) begin
                $display("%3t  | %b | %b | %b | %b |      %b/%b         |     %b/%b       | PASS", 
                    $time, a, b, c, d, expected_out, expected_out_n, out, out_n);
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("%3t  | %b | %b | %b | %b |      %b/%b         |     %b/%b       | FAIL", 
                    $time, a, b, c, d, expected_out, expected_out_n, out, out_n);
            end
            total_tests = total_tests + 1;
        end
        
        // Additional specific test cases with annotations
        $display("\n===============================================================");
        $display("Specific Test Cases:");
        $display("---------------------------------------------------------------");
        
        // Test case where both AND gates output 1
        a = 1'b1; b = 1'b1; c = 1'b1; d = 1'b1;
        #10;
        $display("Both ANDs = 1: a=%b, b=%b, c=%b, d=%b => out=%b, out_n=%b", 
            a, b, c, d, out, out_n);
        
        // Test case where only top AND outputs 1
        a = 1'b1; b = 1'b1; c = 1'b0; d = 1'b1;
        #10;
        $display("Top AND = 1:   a=%b, b=%b, c=%b, d=%b => out=%b, out_n=%b", 
            a, b, c, d, out, out_n);
        
        // Test case where only bottom AND outputs 1
        a = 1'b0; b = 1'b1; c = 1'b1; d = 1'b1;
        #10;
        $display("Bottom AND = 1: a=%b, b=%b, c=%b, d=%b => out=%b, out_n=%b", 
            a, b, c, d, out, out_n);
        
        // Test case where both AND gates output 0
        a = 1'b0; b = 1'b0; c = 1'b0; d = 1'b0;
        #10;
        $display("Both ANDs = 0: a=%b, b=%b, c=%b, d=%b => out=%b, out_n=%b", 
            a, b, c, d, out, out_n);
        
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
        $dumpfile("wire_decl_gate_tb.vcd");
        $dumpvars(0, tb_wire_decl_gate);
    end
    
endmodule