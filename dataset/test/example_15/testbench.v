`timescale 1ns / 1ps

module tb_counter_with_enable;
    
    // Inputs
    reg clk;
    reg rst;
    reg en;
    
    // Output
    wire [3:0] count;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    
    // Clock generation
    always #5 clk = ~clk;  // 10ns clock period
    
    // Instantiate the module under test
    up_counter uut(
        .clk(clk),
        .rst(rst),
        .en(en),
        .count(count)
    );
    
    // Task to check counter value
    task check_count;
        input [3:0] expected_value;
        input [127:0] test_description;
        begin
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Expected: count=%0d", expected_value);
            $display("  Got:      count=%0d", count);
            
            if (count === expected_value) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected %0d but got %0d", expected_value, count);
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 15: Counter with Enable Signal");
        $display("Description: 4-bit up counter (0-15) with enable");
        $display("Features:");
        $display("  - Counts only when enable (en) is high");
        $display("  - Holds value when enable is low");
        $display("  - Resets to 0");
        $display("  - Wraps from 15 to 0");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        en = 0;
        
        // Test 1: Reset behavior
        #20;
        check_count(4'd0, "Reset test - counter should be 0");
        
        // Release reset
        @(negedge clk);
        rst = 0;
        
        // Test 2: Enable = 0, counter should not increment
        $display("Test Set 2: Counter disabled (en=0)");
        en = 0;
        
        @(posedge clk);
        check_count(4'd0, "After 1 clock with en=0");
        
        @(posedge clk);
        check_count(4'd0, "After 2 clocks with en=0");
        
        @(posedge clk);
        check_count(4'd0, "After 3 clocks with en=0");
        
        // Test 3: Enable = 1, counter should increment
        $display("Test Set 3: Counter enabled (en=1)");
        @(negedge clk);
        en = 1;
        
        @(posedge clk);
        check_count(4'd1, "After 1 clock with en=1");
        
        @(posedge clk);
        check_count(4'd2, "After 2 clocks with en=1");
        
        @(posedge clk);
        check_count(4'd3, "After 3 clocks with en=1");
        
        // Test 4: Disable counter mid-count
        $display("Test Set 4: Disable counter mid-count");
        @(negedge clk);
        en = 0;
        
        @(posedge clk);
        check_count(4'd3, "Counter holds at 3 when disabled");
        
        @(posedge clk);
        check_count(4'd3, "Counter still at 3 after 2nd clock");
        
        // Test 5: Re-enable counter
        $display("Test Set 5: Re-enable counter");
        @(negedge clk);
        en = 1;
        
        @(posedge clk);
        check_count(4'd4, "Counter resumes from 4");
        
        @(posedge clk);
        check_count(4'd5, "Counter at 5");
        
        // Test 6: Count to wrap-around
        $display("Test Set 6: Count to wrap-around (15 to 0)");
        // Count up to 15
        for (i = 6; i <= 15; i = i + 1) begin
            @(posedge clk);
            check_count(i[3:0], "Counting up");
        end
        
        // Check wrap to 0
        @(posedge clk);
        check_count(4'd0, "Wrap from 15 to 0");
        
        // Test 7: Enable toggling pattern
        $display("Test Set 7: Enable toggling pattern");
        @(negedge clk);
        en = 0;
        @(posedge clk);
        check_count(4'd0, "en=0, hold at 0");
        
        @(negedge clk);
        en = 1;
        @(posedge clk);
        check_count(4'd1, "en=1, increment to 1");
        
        @(negedge clk);
        en = 0;
        @(posedge clk);
        check_count(4'd1, "en=0, hold at 1");
        
        @(negedge clk);
        en = 1;
        @(posedge clk);
        check_count(4'd2, "en=1, increment to 2");
        
        // Test 8: Reset during counting
        $display("Test Set 8: Reset during counting");
        // Count to some value
        for (i = 3; i <= 7; i = i + 1) begin
            @(posedge clk);
        end
        
        // Apply reset
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        check_count(4'd0, "Reset during counting - should be 0");
        
        @(negedge clk);
        rst = 0;
        en = 1;
        
        // Test 9: Long run with enable always high
        $display("Test Set 9: Full cycle with enable high");
        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            check_count((i + 1) & 4'hF, "Full cycle count");
        end
        
        // Test 10: Enable changes at various times
        $display("Test Set 10: Enable timing tests");
        
        // Enable just before posedge
        @(negedge clk);
        en = 0;
        #4;  // Just before posedge
        en = 1;
        @(posedge clk);
        #1;
        test_count = test_count + 1;
        if (count == 4'd2 || count == 4'd1) begin
            $display("Test %0d: PASS - Enable timing acceptable (count=%0d)", test_count, count);
            pass_count = pass_count + 1;
        end else begin
            $display("Test %0d: FAIL - Unexpected count=%0d", test_count, count);
            test_passed = 1'b0;
        end
        
        // Test 11: Multiple resets
        $display("\nTest Set 11: Multiple resets");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        @(negedge clk);
        rst = 0;
        en = 1;
        
        @(posedge clk);
        check_count(4'd1, "Count after reset release");
        
        @(posedge clk);
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        check_count(4'd0, "Second reset");
        
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