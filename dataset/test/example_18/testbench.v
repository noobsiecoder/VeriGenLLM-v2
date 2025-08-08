`timescale 1ns / 1ps

module tb_abro_fsm;
    
    // Inputs
    reg clk;
    reg rst;
    reg A, B, R;
    
    // Output
    wire O;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Instantiate the module under test
    abro_fsm uut(
        .clk(clk),
        .rst(rst),
        .A(A),
        .B(B),
        .R(R),
        .O(O)
    );
    
    // Task to set inputs and check output
    task set_inputs_and_check;
        input a_val, b_val, r_val;
        input expected_o;
        input [127:0] description;
        begin
            @(negedge clk);
            A = a_val;
            B = b_val;
            R = r_val;
            @(posedge clk);
            #1; // Small delay for output to settle
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, description);
            $display("  Inputs: A=%b, B=%b, R=%b", A, B, R);
            $display("  Expected O=%b, Got O=%b", expected_o, O);
            
            if (O === expected_o) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    // Task to apply sequence
    task apply_sequence;
        input [127:0] description;
        begin
            $display("\n%0s", description);
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 18: ABRO FSM");
        $display("Description: Output O goes high after both A and B");
        $display("have been seen (in any order), reset by R");
        $display("States:");
        $display("  IDLE: Initial state");
        $display("  A_SEEN: A has been detected");
        $display("  B_SEEN: B has been detected");
        $display("  DONE: Both A and B seen, O=1");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        A = 0;
        B = 0;
        R = 0;
        
        // Release reset
        #20;
        @(negedge clk);
        rst = 0;
        
        // Test 1: Basic A then B sequence
        apply_sequence("Test Set 1: A then B sequence");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "Initial state - no inputs");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "A=1, B=0 - waiting for B");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "A=0, B=0 - A was seen");
        set_inputs_and_check(1'b0, 1'b1, 1'b0, 1'b1, "A=0, B=1 - both seen, O=1!");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b1, "Both inputs 0 - O stays 1");
        
        // Test 2: Reset with R
        apply_sequence("Test Set 2: Reset with R signal");
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "R=1 - reset to IDLE");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "R=0 - back in IDLE");
        
        // Test 3: B then A sequence
        apply_sequence("Test Set 3: B then A sequence");
        set_inputs_and_check(1'b0, 1'b1, 1'b0, 1'b0, "B=1, A=0 - waiting for A");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "B=0, A=0 - B was seen");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b1, "A=1, B=0 - both seen, O=1!");
        
        // Test 4: Reset again
        apply_sequence("Test Set 4: Reset from DONE state");
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "R=1 - reset from DONE");
        
        // Test 5: Simultaneous A and B
        apply_sequence("Test Set 5: Simultaneous A and B");
        set_inputs_and_check(1'b1, 1'b1, 1'b0, 1'b1, "A=1, B=1 - immediate DONE");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b1, "A=0, B=0 - stay in DONE");
        
        // Test 6: Reset and test persistence
        apply_sequence("Test Set 6: State persistence test");
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "Reset to IDLE");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "A=1 - go to A_SEEN");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "A=1 again - stay in A_SEEN");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "A=0 - stay in A_SEEN");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "A=1 - still in A_SEEN");
        set_inputs_and_check(1'b1, 1'b1, 1'b0, 1'b1, "A=1, B=1 - finally DONE");
        
        // Test 7: R during different states
        apply_sequence("Test Set 7: R signal in different states");
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "Reset to IDLE");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "Go to A_SEEN");
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "R during A_SEEN - reset");
        set_inputs_and_check(1'b0, 1'b1, 1'b0, 1'b0, "Go to B_SEEN");
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "R during B_SEEN - reset");
        
        // Test 8: Multiple A's before B
        apply_sequence("Test Set 8: Multiple A pulses before B");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "First A");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "A=0");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b0, "Second A");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b0, "A=0");
        set_inputs_and_check(1'b0, 1'b1, 1'b0, 1'b1, "Finally B - output goes high");
        
        // Test 9: R has priority over other inputs
        apply_sequence("Test Set 9: R priority test");
        set_inputs_and_check(1'b1, 1'b1, 1'b1, 1'b0, "A=1, B=1, R=1 - R wins");
        set_inputs_and_check(1'b1, 1'b1, 1'b0, 1'b1, "R=0, A=1, B=1 - now DONE");
        
        // Test 10: Once in DONE, stays in DONE (until R)
        apply_sequence("Test Set 10: DONE state persistence");
        set_inputs_and_check(1'b0, 1'b0, 1'b0, 1'b1, "In DONE, all inputs 0");
        set_inputs_and_check(1'b1, 1'b0, 1'b0, 1'b1, "In DONE, A=1");
        set_inputs_and_check(1'b0, 1'b1, 1'b0, 1'b1, "In DONE, B=1");
        set_inputs_and_check(1'b1, 1'b1, 1'b0, 1'b1, "In DONE, A=1, B=1");
        
        // Final reset
        set_inputs_and_check(1'b0, 1'b0, 1'b1, 1'b0, "Final reset");
        
        // Display test summary
        #10;
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed) begin
            $display("\nRESULT: ALL TESTS PASSED ✓");
        end else begin
            $display("\nRESULT: TESTS FAILED ✗");
        end
        $display("====================================================\n");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #10000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule