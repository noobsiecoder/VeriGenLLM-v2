`timescale 1ns / 1ps

module tb_shift_and_rotate;
    
    // Inputs
    reg [7:0] in;
    reg sel;
    
    // Output
    wire [7:0] out;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    shift_and_rotate uut(
        .in(in),
        .sel(sel),
        .out(out)
    );
    
    // Task to check output
    task check_output;
        input [7:0] test_in;
        input test_sel;
        input [7:0] expected_out;
        input [127:0] test_description;
        begin
            in = test_in;
            sel = test_sel;
            #10; // Wait for combinational logic
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Input:    in=%b (%h), sel=%b", in, in, sel);
            $display("  Expected: out=%b (%h)", expected_out, expected_out);
            $display("  Got:      out=%b (%h)", out, out);
            
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
        $display("Problem 10: Shift Left and Rotate Left");
        $display("Description:");
        $display("  sel=0: Logical shift left (LSL) - insert 0 at LSB");
        $display("  sel=1: Rotate left (ROL) - MSB goes to LSB");
        $display("====================================================\n");
        
        // Test 1: Shift left - basic patterns
        $display("Testing Shift Left (sel=0):");
        
        // Test with pattern 10101010
        check_output(8'b10101010, 1'b0, 8'b01010100, "Shift left: 10101010 -> 01010100");
        
        // Test with all 1s
        check_output(8'b11111111, 1'b0, 8'b11111110, "Shift left: 11111111 -> 11111110");
        
        // Test with all 0s
        check_output(8'b00000000, 1'b0, 8'b00000000, "Shift left: 00000000 -> 00000000");
        
        // Test with single bit set
        check_output(8'b00000001, 1'b0, 8'b00000010, "Shift left: 00000001 -> 00000010");
        check_output(8'b10000000, 1'b0, 8'b00000000, "Shift left: 10000000 -> 00000000 (MSB lost)");
        
        // Test with alternating pattern
        check_output(8'b01010101, 1'b0, 8'b10101010, "Shift left: 01010101 -> 10101010");
        
        // Test 2: Rotate left - basic patterns
        $display("Testing Rotate Left (sel=1):");
        
        // Test with pattern 10101010
        check_output(8'b10101010, 1'b1, 8'b01010101, "Rotate left: 10101010 -> 01010101");
        
        // Test with all 1s
        check_output(8'b11111111, 1'b1, 8'b11111111, "Rotate left: 11111111 -> 11111111 (unchanged)");
        
        // Test with all 0s
        check_output(8'b00000000, 1'b1, 8'b00000000, "Rotate left: 00000000 -> 00000000 (unchanged)");
        
        // Test with single bit set
        check_output(8'b00000001, 1'b1, 8'b00000010, "Rotate left: 00000001 -> 00000010");
        check_output(8'b10000000, 1'b1, 8'b00000001, "Rotate left: 10000000 -> 00000001 (MSB to LSB)");
        
        // Test with alternating pattern
        check_output(8'b01010101, 1'b1, 8'b10101010, "Rotate left: 01010101 -> 10101010");
        
        // Test 3: Edge cases and special patterns
        $display("Testing edge cases:");
        
        // Test shift with MSB set
        check_output(8'b10110011, 1'b0, 8'b01100110, "Shift left: 10110011 -> 01100110");
        
        // Test rotate with MSB set
        check_output(8'b10110011, 1'b1, 8'b01100111, "Rotate left: 10110011 -> 01100111");
        
        // Test sequential pattern
        check_output(8'b00001111, 1'b0, 8'b00011110, "Shift left: 00001111 -> 00011110");
        check_output(8'b00001111, 1'b1, 8'b00011110, "Rotate left: 00001111 -> 00011110");
        
        // Test MSB preservation in rotate
        check_output(8'b11110000, 1'b1, 8'b11100001, "Rotate left: 11110000 -> 11100001");
        
        // Test 4: Verify combinational behavior
        $display("Testing combinational behavior:");
        
        in = 8'b10101010;
        sel = 1'b0;
        #5;
        if (out === 8'b01010100) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Combinational output correct for shift", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Combinational output incorrect", test_count);
            test_passed = 1'b0;
        end
        
        // Change select without changing input
        sel = 1'b1;
        #5;
        if (out === 8'b01010101) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output updates with select change", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Output doesn't update with select", test_count);
            test_passed = 1'b0;
        end
        
        // Test 5: Random patterns
        $display("\nTesting with random patterns:");
        in = 8'b11001100;
        sel = 1'b0;
        #10;
        check_output(8'b11001100, 1'b0, 8'b10011000, "Shift left: 11001100 -> 10011000");
        
        in = 8'b11001100;
        sel = 1'b1;
        #10;
        check_output(8'b11001100, 1'b1, 8'b10011001, "Rotate left: 11001100 -> 10011001");
        
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