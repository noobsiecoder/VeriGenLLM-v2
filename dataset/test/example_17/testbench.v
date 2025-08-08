`timescale 1ns / 1ps

module tb_arithmetic_shift_register_64;
    
    // Inputs
    reg clk;
    reg rst;
    reg shift_left;
    reg enable;
    reg [63:0] data_in;
    reg load;
    
    // Output
    wire [63:0] data_out;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Instantiate the module under test
    arithmetic_shift_register uut(
        .clk(clk),
        .rst(rst),
        .shift_left(shift_left),
        .enable(enable),
        .data_in(data_in),
        .load(load),
        .data_out(data_out)
    );
    
    // Task to check register value
    task check_value;
        input [63:0] expected_value;
        input [127:0] test_description;
        begin
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Expected: %h", expected_value);
            $display("  Got:      %h", data_out);
            
            if (data_out === expected_value) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    // Task to perform shift and check
    task shift_and_check;
        input direction;  // 1 for left, 0 for right
        input [63:0] expected_value;
        input [127:0] description;
        begin
            @(negedge clk);
            shift_left = direction;
            enable = 1'b1;
            @(posedge clk);
            @(negedge clk);
            enable = 1'b0;
            #1;
            
            check_value(expected_value, description);
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 17: 64-bit Arithmetic Shift Register");
        $display("Description:");
        $display("  - Arithmetic shift preserves sign bit on right shift");
        $display("  - Left shift fills with 0");
        $display("  - Load, shift left/right with enable control");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        rst = 1;
        shift_left = 0;
        enable = 0;
        data_in = 64'h0;
        load = 0;
        
        // Test 1: Reset behavior
        #20;
        check_value(64'h0, "Reset test - should be 0");
        
        // Release reset
        @(negedge clk);
        rst = 0;
        
        // Test 2: Load positive value
        $display("Test Set 2: Load and shift positive value");
        @(negedge clk);
        data_in = 64'h0000_0000_0000_00FF;  // 255
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        check_value(64'h0000_0000_0000_00FF, "Load positive value (255)");
        
        // Shift left
        shift_and_check(1'b1, 64'h0000_0000_0000_01FE, "Shift left: 255 << 1 = 510");
        shift_and_check(1'b1, 64'h0000_0000_0000_03FC, "Shift left: 510 << 1 = 1020");
        
        // Shift right
        shift_and_check(1'b0, 64'h0000_0000_0000_01FE, "Shift right: 1020 >> 1 = 510");
        shift_and_check(1'b0, 64'h0000_0000_0000_00FF, "Shift right: 510 >> 1 = 255");
        
        // Test 3: Load negative value (sign extension test)
        $display("Test Set 3: Load and shift negative value");
        @(negedge clk);
        data_in = 64'hFFFF_FFFF_FFFF_FF00;  // -256 in 2's complement
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        check_value(64'hFFFF_FFFF_FFFF_FF00, "Load negative value (-256)");
        
        // Arithmetic shift right (should preserve sign)
        shift_and_check(1'b0, 64'hFFFF_FFFF_FFFF_FF80, "Arithmetic right: -256 >>> 1 = -128");
        shift_and_check(1'b0, 64'hFFFF_FFFF_FFFF_FFC0, "Arithmetic right: -128 >>> 1 = -64");
        
        // Shift left (should shift in 0)
        shift_and_check(1'b1, 64'hFFFF_FFFF_FFFF_FF80, "Shift left: -64 << 1 = -128");
        
        // Test 4: Test enable signal
        $display("Test Set 4: Test enable signal");
        @(negedge clk);
        data_in = 64'h1234_5678_9ABC_DEF0;
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        check_value(64'h1234_5678_9ABC_DEF0, "Load test pattern");
        
        // Try to shift with enable = 0
        @(negedge clk);
        shift_left = 1'b1;
        enable = 1'b0;
        @(posedge clk);
        #1;
        check_value(64'h1234_5678_9ABC_DEF0, "No shift when enable=0");
        
        // Test 5: Multiple shifts
        $display("Test Set 5: Multiple consecutive shifts");
        enable = 1'b1;
        shift_left = 1'b1;
        
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
        end
        @(negedge clk);
        enable = 1'b0;
        #1;
        check_value(64'h2345_678_9ABC_DEF0_0, "After 4 left shifts");
        
        // Test 6: Sign bit preservation
        $display("Test Set 6: Sign bit preservation test");
        @(negedge clk);
        data_in = 64'h8000_0000_0000_0000;  // Minimum negative value
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        check_value(64'h8000_0000_0000_0000, "Load MSB set only");
        
        // Arithmetic shift right
        shift_and_check(1'b0, 64'hC000_0000_0000_0000, "Arithmetic right shift preserves sign");
        shift_and_check(1'b0, 64'hE000_0000_0000_0000, "Another arithmetic right shift");
        
        // Test 7: Edge case - all ones
        $display("Test Set 7: All ones pattern");
        @(negedge clk);
        data_in = 64'hFFFF_FFFF_FFFF_FFFF;
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        check_value(64'hFFFF_FFFF_FFFF_FFFF, "Load all ones (-1)");
        
        shift_and_check(1'b0, 64'hFFFF_FFFF_FFFF_FFFF, "Arithmetic right of -1 stays -1");
        shift_and_check(1'b1, 64'hFFFF_FFFF_FFFF_FFFE, "Left shift of -1");
        
        // Test 8: Maximum positive value
        $display("Test Set 8: Maximum positive value");
        @(negedge clk);
        data_in = 64'h7FFF_FFFF_FFFF_FFFF;  // Maximum positive
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        check_value(64'h7FFF_FFFF_FFFF_FFFF, "Load max positive");
        
        shift_and_check(1'b0, 64'h3FFF_FFFF_FFFF_FFFF, "Right shift max positive");
        shift_and_check(1'b1, 64'h7FFF_FFFF_FFFF_FFFE, "Left shift");
        
        // Test 9: Load during shift (load has priority)
        $display("Test Set 9: Load priority test");
        @(negedge clk);
        data_in = 64'hAAAA_AAAA_AAAA_AAAA;
        load = 1'b1;
        enable = 1'b1;
        shift_left = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        enable = 1'b0;
        check_value(64'hAAAA_AAAA_AAAA_AAAA, "Load has priority over shift");
        
        // Test 10: Pattern shifting
        $display("Test Set 10: Pattern shifting visualization");
        @(negedge clk);
        data_in = 64'h0000_0000_0000_0001;
        load = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load = 1'b0;
        
        // Shift left multiple times to see bit movement
        enable = 1'b1;
        shift_left = 1'b1;
        for (i = 0; i < 8; i = i + 1) begin
            @(posedge clk);
            #1;
            $display("  After %0d left shifts: %h", i+1, data_out);
        end
        @(negedge clk);
        enable = 1'b0;
        
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