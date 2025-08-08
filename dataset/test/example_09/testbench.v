`timescale 1ns / 1ps

module tb_fsm_two_states;
    
    // Inputs
    reg clk;
    reg rst;
    reg in;
    
    // Output
    wire out;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Instantiate the module under test
    fsm_two_states uut(
        .clk(clk),
        .rst(rst),
        .in(in),
        .out(out)
    );
    
    // Task to check FSM output
    task check_fsm;
        input expected_out;
        input [127:0] test_description;
        begin
            #1; // Small delay to let output settle
            test_count = test_count + 1;
            
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Current: in=%b, out=%b", in, out);
            $display("  Expected out=%b", expected_out);
            
            if (out === expected_out) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 9: FSM with Two States");
        $display("Description: Toggle FSM that switches states on input");
        $display("Expected behavior:");
        $display("  - Reset puts FSM in STATE_0 (out=0)");
        $display("  - When in STATE_0: in=1 goes to STATE_1, in=0 stays");
        $display("  - When in STATE_1: in=1 goes to STATE_0, in=0 stays");
        $display("  - Output follows state: STATE_0->out=0, STATE_1->out=1");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        in = 0;
        
        // Test 1: Reset behavior
        #20;
        check_fsm(1'b0, "Reset test - FSM should be in STATE_0 (out=0)");
        
        // Release reset
        @(negedge clk);
        rst = 0;
        
        // Test 2: STATE_0 with in=0 (should stay in STATE_0)
        in = 0;
        @(posedge clk);
        check_fsm(1'b0, "STATE_0 with in=0 - should stay in STATE_0");
        
        // Test 3: STATE_0 with in=1 (should go to STATE_1)
        in = 1;
        @(posedge clk);
        check_fsm(1'b1, "STATE_0 with in=1 - should go to STATE_1");
        
        // Test 4: STATE_1 with in=0 (should stay in STATE_1)
        in = 0;
        @(posedge clk);
        check_fsm(1'b1, "STATE_1 with in=0 - should stay in STATE_1");
        
        // Test 5: STATE_1 with in=1 (should go back to STATE_0)
        in = 1;
        @(posedge clk);
        check_fsm(1'b0, "STATE_1 with in=1 - should go back to STATE_0");
        
        // Test 6: Multiple transitions
        $display("Testing multiple transitions:");
        
        // Currently in STATE_0
        in = 1;
        @(posedge clk);
        check_fsm(1'b1, "Transition 1: STATE_0 -> STATE_1");
        
        in = 1;
        @(posedge clk);
        check_fsm(1'b0, "Transition 2: STATE_1 -> STATE_0");
        
        in = 0;
        @(posedge clk);
        check_fsm(1'b0, "No transition: Stay in STATE_0");
        
        in = 1;
        @(posedge clk);
        check_fsm(1'b1, "Transition 3: STATE_0 -> STATE_1");
        
        // Test 7: Reset during STATE_1
        $display("Testing reset during STATE_1:");
        // Ensure we're in STATE_1
        if (out !== 1'b1) begin
            in = 1;
            @(posedge clk);
        end
        
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        check_fsm(1'b0, "Reset during STATE_1 - should reset to STATE_0");
        
        @(negedge clk);
        rst = 0;
        
        // Test 8: Rapid input changes
        $display("Testing rapid input changes:");
        in = 0;
        @(posedge clk);
        check_fsm(1'b0, "After reset, in=0, stay in STATE_0");
        
        // Toggle input rapidly
        in = 1; @(posedge clk);
        check_fsm(1'b1, "Quick toggle 1: STATE_0 -> STATE_1");
        
        in = 1; @(posedge clk);
        check_fsm(1'b0, "Quick toggle 2: STATE_1 -> STATE_0");
        
        in = 1; @(posedge clk);
        check_fsm(1'b1, "Quick toggle 3: STATE_0 -> STATE_1");
        
        // Test 9: Hold in=0 for multiple cycles
        $display("Testing hold in=0:");
        in = 0;
        @(posedge clk);
        check_fsm(1'b1, "Hold in=0, cycle 1 - stay in STATE_1");
        
        @(posedge clk);
        check_fsm(1'b1, "Hold in=0, cycle 2 - stay in STATE_1");
        
        @(posedge clk);
        check_fsm(1'b1, "Hold in=0, cycle 3 - stay in STATE_1");
        
        // Test 10: Verify output is purely combinational based on state
        $display("Testing output is based on current state:");
        // Go to STATE_0
        in = 1;
        @(posedge clk);
        check_fsm(1'b0, "Back to STATE_0");
        
        // Change input but don't clock - output should not change
        in = 1;
        #2;  // Wait less than clock period
        if (out === 1'b0) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output remains stable between clocks", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Output changed without clock", test_count);
            test_passed = 1'b0;
        end
        
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