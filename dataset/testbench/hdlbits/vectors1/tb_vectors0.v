`timescale 1ns/1ps

module tb_top_module();
    // Inputs (driven by testbench)
    reg [2:0] vec;
    
    // Outputs (driven by DUT)
    wire [2:0] outv;
    wire o2, o1, o0;
    
    // Instantiate the Design Under Test (DUT)
    top_module dut (
        .vec(vec),
        .outv(outv),
        .o2(o2),
        .o1(o1),
        .o0(o0)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    // Expected values
    reg [2:0] expected_outv;
    reg expected_o2, expected_o1, expected_o0;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing top_module (Vector Operations)");
        $display("Function: outv = vec, o0 = vec[0], o1 = vec[1], o2 = vec[2]");
        $display("===============================================================");
        $display("Time | vec[2:0] | Expected outv | Actual outv | Expected o2,o1,o0 | Actual o2,o1,o0 | Result");
        $display("----------------------------------------------------------------------------------------");
        
        // Test all 8 possible 3-bit combinations
        for (i = 0; i < 8; i = i + 1) begin
            vec = i[2:0];
            
            // Calculate expected values
            expected_outv = vec;
            expected_o0 = vec[0];
            expected_o1 = vec[1];
            expected_o2 = vec[2];
            
            // Wait for propagation
            #10;
            
            // Check all outputs
            if (outv == expected_outv && o0 == expected_o0 && o1 == expected_o1 && o2 == expected_o2) begin
                $display("%3t  |   %b    |      %b      |     %b     |      %b,%b,%b      |     %b,%b,%b     | PASS", 
                    $time, vec, expected_outv, outv, expected_o2, expected_o1, expected_o0, o2, o1, o0);
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("%3t  |   %b    |      %b      |     %b     |      %b,%b,%b      |     %b,%b,%b     | FAIL", 
                    $time, vec, expected_outv, outv, expected_o2, expected_o1, expected_o0, o2, o1, o0);
            end
            total_tests = total_tests + 1;
        end
        
        // Additional detailed test cases with annotations
        $display("\n===============================================================");
        $display("Detailed Test Cases:");
        $display("---------------------------------------------------------------");
        
        // Test case 1: All zeros
        vec = 3'b000;
        #10;
        $display("All zeros: vec=%b => outv=%b, o2=%b, o1=%b, o0=%b", 
            vec, outv, o2, o1, o0);
        
        // Test case 2: All ones
        vec = 3'b111;
        #10;
        $display("All ones:  vec=%b => outv=%b, o2=%b, o1=%b, o0=%b", 
            vec, outv, o2, o1, o0);
        
        // Test case 3: Alternating pattern
        vec = 3'b101;
        #10;
        $display("Alternate: vec=%b => outv=%b, o2=%b, o1=%b, o0=%b", 
            vec, outv, o2, o1, o0);
        
        // Test case 4: Walking 1 pattern
        $display("\nWalking 1 Test:");
        vec = 3'b001;
        #10;
        $display("  vec=%b => outv=%b, o2=%b, o1=%b, o0=%b (only LSB set)", 
            vec, outv, o2, o1, o0);
        
        vec = 3'b010;
        #10;
        $display("  vec=%b => outv=%b, o2=%b, o1=%b, o0=%b (only middle bit set)", 
            vec, outv, o2, o1, o0);
        
        vec = 3'b100;
        #10;
        $display("  vec=%b => outv=%b, o2=%b, o1=%b, o0=%b (only MSB set)", 
            vec, outv, o2, o1, o0);
        
        // Test case 5: Binary counting visualization
        $display("\nBinary Counting Sequence:");
        $display("Decimal | Binary | Vector | Individual Bits");
        $display("--------|--------|--------|----------------");
        for (i = 0; i < 8; i = i + 1) begin
            vec = i[2:0];
            #10;
            $display("   %0d    |  %b   |  %b   | o2=%b, o1=%b, o0=%b", 
                i, i[2:0], outv, o2, o1, o0);
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
        $dumpfile("top_module_vectors_tb.vcd");
        $dumpvars(0, tb_top_module);
    end
    
endmodule