`timescale 1ns/1ps

module tb_top_module_always_nolatches();
    // Input
    reg [15:0] scancode;
    
    // Outputs
    wire left, down, right, up;
    
    // Instantiate DUT
    top_module_always_nolatches dut (
        .scancode(scancode),
        .left(left),
        .down(down),
        .right(right),
        .up(up)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing PS/2 Keyboard Scancode Decoder");
        $display("Recognizes arrow key scancodes");
        $display("===============================================================");
        
        // Test 1: Valid arrow key scancodes
        $display("\nTest 1: Valid Arrow Key Scancodes");
        $display("---------------------------------------------------------------");
        $display("Scancode | left | down | right | up | Result");
        $display("---------------------------------------------------------------");
        
        // Test left arrow
        scancode = 16'he06b;
        #10;
        check_arrow_key("Left arrow", 1'b1, 1'b0, 1'b0, 1'b0);
        
        // Test down arrow
        scancode = 16'he072;
        #10;
        check_arrow_key("Down arrow", 1'b0, 1'b1, 1'b0, 1'b0);
        
        // Test right arrow
        scancode = 16'he074;
        #10;
        check_arrow_key("Right arrow", 1'b0, 1'b0, 1'b1, 1'b0);
        
        // Test up arrow
        scancode = 16'he075;
        #10;
        check_arrow_key("Up arrow", 1'b0, 1'b0, 1'b0, 1'b1);
        
        // Test 2: Invalid scancodes (should output all zeros)
        $display("\nTest 2: Invalid Scancodes (No Arrow Keys)");
        $display("---------------------------------------------------------------");
        
        // Test various invalid scancodes
        test_invalid_scancode(16'h0000, "All zeros");
        test_invalid_scancode(16'hFFFF, "All ones");
        test_invalid_scancode(16'he06a, "One bit off from left");
        test_invalid_scancode(16'he06c, "One bit off from left");
        test_invalid_scancode(16'he073, "Between down and right");
        test_invalid_scancode(16'h1234, "Random value");
        test_invalid_scancode(16'he070, "Close to valid range");
        test_invalid_scancode(16'he076, "Just after up");
        
        // Test 3: Sequential scanning
        $display("\nTest 3: Sequential Key Press Simulation");
        $display("---------------------------------------------------------------");
        
        // Simulate pressing keys in sequence
        $display("Simulating key sequence: Left -> Down -> Right -> Up -> None");
        
        scancode = 16'he06b; #10;
        $display("  Left pressed: L=%b D=%b R=%b U=%b", left, down, right, up);
        
        scancode = 16'he072; #10;
        $display("  Down pressed: L=%b D=%b R=%b U=%b", left, down, right, up);
        
        scancode = 16'he074; #10;
        $display("  Right pressed: L=%b D=%b R=%b U=%b", left, down, right, up);
        
        scancode = 16'he075; #10;
        $display("  Up pressed: L=%b D=%b R=%b U=%b", left, down, right, up);
        
        scancode = 16'h0000; #10;
        $display("  No key: L=%b D=%b R=%b U=%b", left, down, right, up);
        
        // Test 4: Rapid transitions
        $display("\nTest 4: Rapid Key Transitions");
        $display("---------------------------------------------------------------");
        
        $display("Testing rapid transitions between keys:");
        scancode = 16'he06b; #2;  // Left
        scancode = 16'he074; #2;  // Right
        scancode = 16'he072; #2;  // Down
        scancode = 16'he075; #2;  // Up
        scancode = 16'h0000; #2;  // None
        
        if (left == 0 && down == 0 && right == 0 && up == 0) begin
            $display("Final state correct (all zeros): PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Final state incorrect: FAIL");
        end
        total_tests = total_tests + 1;
        
        // Test 5: Boundary values around valid scancodes
        $display("\nTest 5: Boundary Value Testing");
        $display("---------------------------------------------------------------");
        
        // Test values just before and after valid scancodes
        test_boundary(16'he06a, 16'he06b, 16'he06c, "left");
        test_boundary(16'he071, 16'he072, 16'he073, "down");
        test_boundary(16'he073, 16'he074, 16'he075, "right");
        test_boundary(16'he074, 16'he075, 16'he076, "up");
        
        // Test 6: Latch prevention verification
        $display("\nTest 6: Latch Prevention Verification");
        $display("---------------------------------------------------------------");
        
        // Set a valid key
        scancode = 16'he06b;  // Left
        #10;
        $display("Set left key: L=%b", left);
        
        // Change to invalid - should clear all outputs
        scancode = 16'h0000;
        #10;
        
        if (left == 0 && down == 0 && right == 0 && up == 0) begin
            $display("All outputs cleared correctly - No latches: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Outputs not cleared - Possible latch: FAIL");
        end
        total_tests = total_tests + 1;
        
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
    
    // Task to check arrow key output
    task check_arrow_key;
        input [20*8:1] key_name;
        input exp_left, exp_down, exp_right, exp_up;
        begin
            total_tests = total_tests + 1;
            
            $display(" %04h   |  %b   |  %b   |   %b   | %b  | %s %s",
                scancode, left, down, right, up, key_name,
                (left == exp_left && down == exp_down && 
                 right == exp_right && up == exp_up) ? "PASS" : "FAIL");
            
            if (left == exp_left && down == exp_down && 
                right == exp_right && up == exp_up) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Task to test invalid scancode
    task test_invalid_scancode;
        input [15:0] test_code;
        input [20*8:1] description;
        begin
            scancode = test_code;
            #10;
            
            total_tests = total_tests + 1;
            
            if (left == 0 && down == 0 && right == 0 && up == 0) begin
                num_tests_passed = num_tests_passed + 1;
                $display(" %04h: %s - All outputs zero: PASS", test_code, description);
            end else begin
                $display(" %04h: %s - L=%b D=%b R=%b U=%b: FAIL", 
                    test_code, description, left, down, right, up);
            end
        end
    endtask
    
    // Task to test boundary values
    task test_boundary;
        input [15:0] before, valid, after;
        input [10*8:1] key_name;
        begin
            // Test value before
            scancode = before;
            #10;
            if (left == 0 && down == 0 && right == 0 && up == 0)
                $display("  %04h (before %0s): Correctly ignored", before, key_name);
            else
                $display("  %04h (before %0s): ERROR - output not zero", before, key_name);
            
            // Test valid value
            scancode = valid;
            #10;
            $display("  %04h (valid %0s): L=%b D=%b R=%b U=%b", 
                valid, key_name, left, down, right, up);
            
            // Test value after
            scancode = after;
            #10;
            if (left == 0 && down == 0 && right == 0 && up == 0)
                $display("  %04h (after %0s): Correctly ignored", after, key_name);
            else
                $display("  %04h (after %0s): ERROR - output not zero", after, key_name);
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("scancode_decoder_tb.vcd");
        $dumpvars(0, tb_top_module_always_nolatches);
    end
    
endmodule