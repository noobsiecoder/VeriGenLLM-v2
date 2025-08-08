`timescale 1ns / 1ps

module tb_counter_1_to_12;
    
    // Inputs
    reg clk;
    reg rst;
    
    // Output
    wire [3:0] count;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    integer expected;
    
    // Clock generation
    always #5 clk = ~clk;  // 10ns clock period
    
    // Instantiate the module under test
    counter_1_to_12 uut(
        .clk(clk),
        .rst(rst),
        .count(count)
    );
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 7: 1-to-12 Counter");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        
        // Wait for reset to take effect
        #20;
        
        // Test 1: Reset behavior
        test_count = test_count + 1;
        if (count === 4'd1) begin
            $display("Test %0d: PASS - Reset sets counter to 1", test_count);
            pass_count = pass_count + 1;
        end else begin
            $display("Test %0d: FAIL - Reset: expected 1, got %0d", test_count, count);
            test_passed = 1'b0;
        end
        
        // Release reset on negedge to avoid race conditions
        @(negedge clk);
        rst = 0;
        
        // Wait for next posedge
        @(posedge clk);
        
        // Test 2: Count from 1 to 12
        $display("\nTesting count sequence:");
        for (i = 1; i <= 12; i = i + 1) begin
            test_count = test_count + 1;
            if (count === i) begin
                $display("  Count %0d: PASS", i);
                pass_count = pass_count + 1;
            end else begin
                $display("  Count %0d: FAIL - expected %0d, got %0d", i, i, count);
                test_passed = 1'b0;
            end
            
            if (i < 12) @(posedge clk);
        end
        
        // Test 3: Wrap around from 12 to 1
        @(posedge clk);
        test_count = test_count + 1;
        if (count === 4'd1) begin
            $display("\nWrap test: PASS - Counter wrapped from 12 to 1");
            pass_count = pass_count + 1;
        end else begin
            $display("\nWrap test: FAIL - expected 1 after 12, got %0d", count);
            test_passed = 1'b0;
        end
        
        // Test 4: Continue counting after wrap
        $display("\nTesting continued counting:");
        for (i = 2; i <= 5; i = i + 1) begin
            @(posedge clk);
            test_count = test_count + 1;
            if (count === i) begin
                $display("  Count %0d: PASS", i);
                pass_count = pass_count + 1;
            end else begin
                $display("  Count %0d: FAIL - expected %0d, got %0d", i, i, count);
                test_passed = 1'b0;
            end
        end
        
        // Test 5: Reset during counting
        $display("\nTesting reset during counting:");
        @(negedge clk);
        rst = 1;
        @(posedge clk);
        #1; // Small delay to let reset propagate
        
        test_count = test_count + 1;
        if (count === 4'd1) begin
            $display("Reset during count: PASS - Counter reset to 1");
            pass_count = pass_count + 1;
        end else begin
            $display("Reset during count: FAIL - expected 1, got %0d", count);
            test_passed = 1'b0;
        end
        
        // Release reset and verify counting resumes
        @(negedge clk);
        rst = 0;
        
        // Test a full cycle
        $display("\nTesting full cycle:");
        for (i = 0; i < 13; i = i + 1) begin
            @(posedge clk);
            expected = (i == 0) ? 1 : ((i == 12) ? 1 : (i % 12) + 1);
            
            test_count = test_count + 1;
            if (count === expected) begin
                $display("  Cycle position %0d: PASS (count=%0d)", i, count);
                pass_count = pass_count + 1;
            end else begin
                $display("  Cycle position %0d: FAIL - expected %0d, got %0d", i, expected, count);
                test_passed = 1'b0;
            end
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