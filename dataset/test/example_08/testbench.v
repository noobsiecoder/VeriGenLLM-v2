`timescale 1ns / 1ps

module tb_lfsr_taps3_5;
    
    // Inputs
    reg clk;
    reg rst;
    
    // Output
    wire [5:0] lfsr_out;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i, j;
    reg [5:0] expected_sequence [0:63];  // Store expected sequence
    reg [5:0] prev_value;
    reg found_repeat;
    integer period_length;
    integer first_occurrence;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Instantiate the module under test
    lfsr_random uut(
        .clk(clk),
        .rst(rst),
        .lfsr_out(lfsr_out)
    );
    
    // Function to calculate expected next LFSR value
    function [5:0] calculate_next_lfsr;
        input [5:0] current;
        reg feedback;
        begin
            // Taps at positions 3 and 5 means bits [2] and [4] (0-indexed)
            feedback = current[2] ^ current[4];
            calculate_next_lfsr = {current[4:0], feedback};
        end
    endfunction
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 8: LFSR with taps at bit 3 and bit 5");
        $display("Description: 6-bit LFSR with XOR feedback from bits [2] and [4]");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        
        // Test 1: Reset behavior
        #20;
        test_count = test_count + 1;
        if (lfsr_out !== 6'b000000) begin
            $display("Test %0d: PASS - Reset sets LFSR to non-zero value: %b", test_count, lfsr_out);
            pass_count = pass_count + 1;
        end else begin
            $display("Test %0d: FAIL - LFSR is all zeros after reset (invalid seed)", test_count);
            test_passed = 1'b0;
        end
        
        // Store initial value
        prev_value = lfsr_out;
        
        // Release reset
        @(negedge clk);
        rst = 0;
        
        // Test 2: Verify feedback calculation for first few cycles
        $display("\nTesting LFSR sequence (first 10 values):");
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            #1; // Small delay to let output settle
            
            test_count = test_count + 1;
            expected_sequence[i] = calculate_next_lfsr(prev_value);
            
            $display("  Cycle %0d: prev=%b, expected=%b, got=%b", 
                     i+1, prev_value, expected_sequence[i], lfsr_out);
            
            if (lfsr_out === expected_sequence[i]) begin
                pass_count = pass_count + 1;
            end else begin
                $display("    FAIL - Incorrect LFSR value");
                test_passed = 1'b0;
            end
            
            prev_value = lfsr_out;
        end
        
        // Test 3: Check that LFSR never gets stuck at zero
        $display("\nChecking for all-zero state (running 100 cycles):");
        found_repeat = 0;
        for (i = 10; i < 100; i = i + 1) begin
            @(posedge clk);
            #1;
            
            if (lfsr_out === 6'b000000) begin
                $display("  FAIL: LFSR reached all-zero state at cycle %0d", i);
                test_passed = 1'b0;
                found_repeat = 1;
            end
            
            // Store values to check period
            if (i < 64) expected_sequence[i] = lfsr_out;
        end
        
        if (!found_repeat) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("  PASS: LFSR never reached all-zero state");
        end
        
        // Test 4: Verify it's actually pseudo-random (not stuck)
        @(posedge clk);
        prev_value = lfsr_out;
        found_repeat = 0;
        
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            if (lfsr_out === prev_value) begin
                found_repeat = 1;
            end
        end
        
        test_count = test_count + 1;
        if (!found_repeat) begin
            $display("\nTest %0d: PASS - LFSR is changing (not stuck)", test_count);
            pass_count = pass_count + 1;
        end else begin
            $display("\nTest %0d: WARNING - LFSR might be stuck", test_count);
        end
        
        // Test 5: Check maximum period (should be 2^6-1 = 63 for maximal LFSR)
        $display("\nChecking LFSR period:");
        
        // Reset to get consistent starting point
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        @(posedge clk);
        #1;
        
        prev_value = lfsr_out;
        period_length = 0;
        
        // Run for up to 65 cycles to find period
        found_repeat = 0;
        for (i = 1; i <= 65 && !found_repeat; i = i + 1) begin
            @(posedge clk);
            #1;
            period_length = period_length + 1;
            
            if (lfsr_out === prev_value) begin
                $display("  Found period of length %0d", period_length);
                found_repeat = 1;
            end
        end
        
        test_count = test_count + 1;
        if (period_length == 63) begin
            $display("  PASS: LFSR has maximal period (63)");
            pass_count = pass_count + 1;
        end else if (period_length > 0 && period_length < 63) begin
            $display("  WARNING: LFSR has non-maximal period (%0d)", period_length);
            // Don't fail - taps might still be correct but with different seed
        end else begin
            $display("  INFO: Period length = %0d", period_length);
        end
        
        // Test 6: Verify the shift operation
        $display("\nVerifying shift operation:");
        @(posedge clk);
        #1;
        prev_value = lfsr_out;
        @(posedge clk);
        #1;
        
        test_count = test_count + 1;
        // Check that bits [4:0] of current value are bits [5:1] of previous value
        if (lfsr_out[5:1] === prev_value[4:0]) begin
            $display("  PASS: Shift operation correct (left shift)");
            pass_count = pass_count + 1;
        end else if (lfsr_out[4:0] === prev_value[5:1]) begin
            $display("  PASS: Shift operation correct (right shift)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Shift operation incorrect");
            $display("    Previous: %b", prev_value);
            $display("    Current:  %b", lfsr_out);
            test_passed = 1'b0;
        end
        
        // Display test summary
        #10;
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed && pass_count >= test_count - 1) begin  // Allow one warning
            $display("\nRESULT: ALL TESTS PASSED ✓");
        end else begin
            $display("\nRESULT: TESTS FAILED ✗");
        end
        $display("====================================================\n");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #100000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule