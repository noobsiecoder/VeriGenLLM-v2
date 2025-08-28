`timescale 1ns/1ps

module tb_top_module_always();
    // Inputs
    reg clk;
    reg a;
    reg b;
    
    // Outputs
    wire out_assign;
    wire out_always_comb;
    wire out_always_ff;
    
    // Instantiate DUT
    top_module_always dut (
        .clk(clk),
        .a(a),
        .b(b),
        .out_assign(out_assign),
        .out_always_comb(out_always_comb),
        .out_always_ff(out_always_ff)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg expected_xor;
    reg expected_ff;
    reg prev_xor;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 10ns period
    end
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        a = 0;
        b = 0;
        prev_xor = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing XOR Gate: Three Implementation Methods");
        $display("1. Continuous assignment (immediate)");
        $display("2. Combinational always (immediate)");
        $display("3. Sequential always (delayed by one clock)");
        $display("===============================================================");
        
        // Wait for initial state to settle
        #20;
        
        // Test 1: Truth table verification
        $display("\nTest 1: XOR Truth Table");
        $display("---------------------------------------------------------------");
        $display("a | b | assign | comb | Expected | Result");
        $display("---------------------------------------------------------------");
        
        // Test all 4 combinations
        for (i = 0; i < 4; i = i + 1) begin
            {a, b} = i[1:0];
            expected_xor = a ^ b;
            #10;  // Wait for combinational logic
            
            total_tests = total_tests + 1;
            
            if (out_assign == expected_xor && out_always_comb == expected_xor) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%b | %b |   %b    |  %b   |    %b     | PASS",
                    a, b, out_assign, out_always_comb, expected_xor);
            end else begin
                $display("%b | %b |   %b    |  %b   |    %b     | FAIL",
                    a, b, out_assign, out_always_comb, expected_xor);
            end
        end
        
        // Test 2: Timing differences
        $display("\nTest 2: Timing Difference Demonstration");
        $display("---------------------------------------------------------------");
        
        // Reset to known state
        a = 0; b = 0;
        @(posedge clk);
        #1;
        
        $display("Initial: a=0, b=0");
        $display("Outputs: assign=%b, comb=%b, ff=%b", 
            out_assign, out_always_comb, out_always_ff);
        
        // Change inputs mid-cycle
        #3;
        a = 1; b = 1;
        expected_xor = a ^ b;  // Should be 0
        #1;
        
        $display("\nMid-cycle change to a=1, b=1:");
        $display("Combinational outputs: assign=%b, comb=%b (should be %b)", 
            out_assign, out_always_comb, expected_xor);
        $display("FF output: %b (still shows old value)", out_always_ff);
        
        total_tests = total_tests + 1;
        if (out_assign == expected_xor && out_always_comb == expected_xor) begin
            $display("Combinational outputs updated immediately: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Combinational outputs incorrect: FAIL");
        end
        
        // Wait for next clock edge
        @(posedge clk);
        #1;
        
        $display("\nAfter clock edge:");
        $display("FF output: %b (should now be %b)", out_always_ff, expected_xor);
        
        total_tests = total_tests + 1;
        if (out_always_ff == expected_xor) begin
            $display("FF updated correctly: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("FF update failed: FAIL");
        end
        
        // Test 3: Sequential behavior verification
        $display("\nTest 3: Sequential Behavior of FF Output");
        $display("---------------------------------------------------------------");
        $display("FF output is delayed by one clock cycle");
        $display("\nTime  | a | b | a^b | assign | comb | ff | Result");
        $display("--------------------------------------------------");
        
        // Set known starting state
        a = 0; b = 0;
        @(posedge clk);
        @(posedge clk);  // Let FF settle to known state
        
        // Test sequence
        for (i = 0; i < 8; i = i + 1) begin
            // Set new inputs at beginning of cycle
            a = $random & 1;
            b = $random & 1;
            expected_xor = a ^ b;
            
            #1;  // Small delay to let combinational outputs settle
            
            $display("%3t | %b | %b |  %b  |   %b    |  %b   | %b  | Comb %s",
                $time, a, b, expected_xor, out_assign, out_always_comb, out_always_ff,
                (out_assign == expected_xor && out_always_comb == expected_xor) ? "OK" : "FAIL");
            
            // Verify combinational outputs
            total_tests = total_tests + 1;
            if (out_assign == expected_xor && out_always_comb == expected_xor) begin
                num_tests_passed = num_tests_passed + 1;
            end
            
            // Wait for next clock edge
            @(posedge clk);
        end
        
        // Test 4: FF delay verification
        $display("\nTest 4: Flip-Flop One-Cycle Delay");
        $display("---------------------------------------------------------------");

        // Clear to known state
        a = 0; b = 0;
        @(posedge clk);
        @(posedge clk);  // Ensure FF is in known state

        // Test sequence showing FF captures input at clock edge
        for (i = 0; i < 5; i = i + 1) begin
            // Set inputs early in the cycle
            a = i & 1;
            b = (i >> 1) & 1;
            expected_xor = a ^ b;
            
            $display("\nCycle %0d:", i);
            $display("  Inputs set to: a=%b, b=%b (XOR=%b)", a, b, expected_xor);
            
            // Wait until just before clock
            #4;
            $display("  Before clock edge: FF output=%b", out_always_ff);
            
            // Clock edge occurs here - FF will capture current XOR value
            @(posedge clk);
            #1;
            
            // Now FF should show the XOR value that was just captured
            $display("  After clock edge: FF output=%b", out_always_ff);
            
            total_tests = total_tests + 1;
            if (out_always_ff == expected_xor) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  FF correctly captured XOR value at clock edge: PASS");
            end else begin
                $display("  FF did not capture correct value: FAIL");
            end
        end
        
        // Test 5: Glitch immunity
        $display("\nTest 5: Flip-Flop Glitch Immunity");
        $display("---------------------------------------------------------------");
        
        a = 0; b = 0;
        @(posedge clk);
        #2;
        
        $display("Creating glitches between clocks:");
        expected_ff = out_always_ff;
        
        // Create multiple transitions
        repeat(10) begin
            a = ~a;
            #0.5;
            $display("  a=%b, b=%b: assign=%b, comb=%b, ff=%b", 
                a, b, out_assign, out_always_comb, out_always_ff);
        end
        
        total_tests = total_tests + 1;
        if (out_always_ff == expected_ff) begin
            $display("FF unchanged by glitches: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("FF affected by glitches: FAIL");
        end
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else
            $display("Overall Result: SOME TESTS PASSED ⚠");
        $display("===============================================================");
        
        $finish;
    end
    
    // Generate VCD file
    initial begin
        $dumpfile("xor_three_ways_tb.vcd");
        $dumpvars(0, tb_top_module_always);
    end
    
endmodule