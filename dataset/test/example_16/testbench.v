`timescale 1ns / 1ps

module tb_fsm_detect_101;
    
    // Inputs
    reg clk;
    reg rst;
    reg in;
    
    // Output
    wire detected;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Instantiate the module under test
    sequence_detector_101 uut(
        .clk(clk),
        .rst(rst),
        .in(in),
        .detected(detected)
    );
    
    // Task to send a bit and check detection
    task send_bit_and_check;
        input bit_value;
        input expected_detected;
        input [127:0] description;
        begin
            @(negedge clk);
            in = bit_value;
            @(posedge clk);
            #1; // Small delay to let output settle
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, description);
            $display("  Input bit: %b, Expected detected: %b, Got: %b", 
                     bit_value, expected_detected, detected);
            
            if (detected === expected_detected) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
            end
        end
    endtask
    
    // Task to send a sequence of bits
    task send_sequence;
        input [31:0] sequence;
        input [4:0] length;
        input [127:0] description;
        integer j;
        begin
            $display("\nSending sequence: %0s", description);
            for (j = length-1; j >= 0; j = j - 1) begin
                @(negedge clk);
                in = sequence[j];
                @(posedge clk);
                #1;
                $display("  Bit %0d: in=%b, detected=%b", length-j, sequence[j], detected);
            end
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 16: FSM to Detect Pattern '101'");
        $display("Description: Serial pattern detector");
        $display("Output goes high when pattern '101' is detected");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        in = 0;
        
        // Release reset
        #20;
        @(negedge clk);
        rst = 0;
        
        // Test 1: Basic pattern detection
        $display("Test Set 1: Basic pattern '101' detection");
        send_bit_and_check(1'b1, 1'b0, "First '1' - no detection yet");
        send_bit_and_check(1'b0, 1'b0, "Got '10' - no detection yet");
        send_bit_and_check(1'b1, 1'b1, "Got '101' - DETECTED!");
        
        // Test 2: Continue after detection
        $display("\nTest Set 2: Continue after detection");
        send_bit_and_check(1'b0, 1'b0, "Got '0' after detection");
        send_bit_and_check(1'b1, 1'b0, "Got '01' - no detection");
        
        // Test 3: Reset and test again
        $display("\nTest Set 3: Reset and test again");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        send_bit_and_check(1'b1, 1'b0, "After reset: first '1'");
        send_bit_and_check(1'b0, 1'b0, "Got '10'");
        send_bit_and_check(1'b1, 1'b1, "Got '101' - DETECTED!");
        
        // Test 4: Multiple consecutive 1s
        $display("\nTest Set 4: Multiple consecutive 1s");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        send_bit_and_check(1'b1, 1'b0, "First '1'");
        send_bit_and_check(1'b1, 1'b0, "Second '1' (11)");
        send_bit_and_check(1'b1, 1'b0, "Third '1' (111)");
        send_bit_and_check(1'b0, 1'b0, "Got '0' (1110)");
        send_bit_and_check(1'b1, 1'b1, "Got '1' (11101) - DETECTED!");
        
        // Test 5: Pattern with leading zeros
        $display("\nTest Set 5: Pattern with leading zeros");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        send_bit_and_check(1'b0, 1'b0, "Leading '0'");
        send_bit_and_check(1'b0, 1'b0, "Another '0'");
        send_bit_and_check(1'b1, 1'b0, "Got '1' (001)");
        send_bit_and_check(1'b0, 1'b0, "Got '0' (0010)");
        send_bit_and_check(1'b1, 1'b1, "Got '1' (00101) - DETECTED!");
        
        // Test 6: Overlapping patterns
        $display("\nTest Set 6: Overlapping patterns");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        // Send 10101 - should detect twice
        send_bit_and_check(1'b1, 1'b0, "First '1'");
        send_bit_and_check(1'b0, 1'b0, "Got '10'");
        send_bit_and_check(1'b1, 1'b1, "Got '101' - First detection!");
        send_bit_and_check(1'b0, 1'b0, "Got '0' (1010)");
        send_bit_and_check(1'b1, 1'b1, "Got '1' (10101) - Second detection!");
        
        // Test 7: Almost patterns
        $display("\nTest Set 7: Almost patterns (100, 111, 001)");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        // 100 - not a match
        send_bit_and_check(1'b1, 1'b0, "Start with '1'");
        send_bit_and_check(1'b0, 1'b0, "Got '10'");
        send_bit_and_check(1'b0, 1'b0, "Got '100' - no detection");
        
        // Test 8: Long sequence test
        $display("\nTest Set 8: Long sequence with multiple patterns");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        // Sequence: 11010110101
        send_sequence(32'b11010110101, 11, "11010110101");
        
        // Test 9: Rapid input changes
        $display("\nTest Set 9: Rapid input changes");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        
        // Quick 101
        in = 1'b1; @(posedge clk); #1;
        if (detected !== 1'b0) begin
            $display("  FAIL: Detected too early at step 1");
            test_passed = 1'b0;
        end
        
        in = 1'b0; @(posedge clk); #1;
        if (detected !== 1'b0) begin
            $display("  FAIL: Detected too early at step 2");
            test_passed = 1'b0;
        end
        
        in = 1'b1; @(posedge clk); #1;
        if (detected !== 1'b1) begin
            $display("  FAIL: Should detect 101 now");
            test_passed = 1'b0;
        end else begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Rapid 101 detected correctly", test_count);
        end
        
        // Test 10: Check detection is single cycle
        $display("\nTest Set 10: Detection pulse width");
        in = 1'b0; // Change input
        @(posedge clk); #1;
        if (detected !== 1'b0) begin
            $display("  FAIL: Detection should be single cycle");
            test_passed = 1'b0;
        end else begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Detection is single cycle", test_count);
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