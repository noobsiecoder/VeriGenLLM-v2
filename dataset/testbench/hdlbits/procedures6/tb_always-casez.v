`timescale 1ns/1ps

module tb_top_module_always_case();
    // Input
    reg [3:0] in;
    
    // Output
    wire [1:0] pos;
    
    // Instantiate DUT
    top_module_always_case dut (
        .in(in),
        .pos(pos)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [1:0] expected_pos;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 4-bit Priority Encoder");
        $display("Priority: bit 3 (highest) to bit 0 (lowest)");
        $display("Output: position of highest set bit, 00 if none");
        $display("===============================================================");
        
        // Test 1: Complete truth table
        $display("\nTest 1: Complete Truth Table (all 16 combinations)");
        $display("---------------------------------------------------------------");
        $display("in[3:0] | Binary | pos | Expected | Result");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 16; i = i + 1) begin
            in = i[3:0];
            
            // Calculate expected position
            if (in[3])
                expected_pos = 2'b11;
            else if (in[2])
                expected_pos = 2'b10;
            else if (in[1])
                expected_pos = 2'b01;
            else if (in[0])
                expected_pos = 2'b00;
            else
                expected_pos = 2'b00;  // No bits set
            
            #10;
            
            total_tests = total_tests + 1;
            if (pos == expected_pos) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  %04b  |   %2d   | %02b  |    %02b    | PASS", 
                    in, i, pos, expected_pos);
            end else begin
                $display("  %04b  |   %2d   | %02b  |    %02b    | FAIL", 
                    in, i, pos, expected_pos);
            end
        end
        
        // Test 2: Priority verification
        $display("\nTest 2: Priority Verification");
        $display("---------------------------------------------------------------");
        
        // Single bit set cases
        in = 4'b0001; #10;
        $display("Only bit 0 set: in=%04b, pos=%02b (expect 00)", in, pos);
        check_result(2'b00, "Bit 0 only");
        
        in = 4'b0010; #10;
        $display("Only bit 1 set: in=%04b, pos=%02b (expect 01)", in, pos);
        check_result(2'b01, "Bit 1 only");
        
        in = 4'b0100; #10;
        $display("Only bit 2 set: in=%04b, pos=%02b (expect 10)", in, pos);
        check_result(2'b10, "Bit 2 only");
        
        in = 4'b1000; #10;
        $display("Only bit 3 set: in=%04b, pos=%02b (expect 11)", in, pos);
        check_result(2'b11, "Bit 3 only");
        
        // Test 3: Multiple bits set (priority test)
        $display("\nTest 3: Multiple Bits Set (Priority Test)");
        $display("---------------------------------------------------------------");
        
        in = 4'b1111; #10;
        $display("All bits set: in=%04b, pos=%02b (expect 11 - highest priority)", in, pos);
        check_result(2'b11, "All bits");
        
        in = 4'b0111; #10;
        $display("Bits 2,1,0 set: in=%04b, pos=%02b (expect 10)", in, pos);
        check_result(2'b10, "Bits 2,1,0");
        
        in = 4'b0011; #10;
        $display("Bits 1,0 set: in=%04b, pos=%02b (expect 01)", in, pos);
        check_result(2'b01, "Bits 1,0");
        
        in = 4'b1010; #10;
        $display("Bits 3,1 set: in=%04b, pos=%02b (expect 11)", in, pos);
        check_result(2'b11, "Bits 3,1");
        
        // Test 4: Zero input case
        $display("\nTest 4: Zero Input Case");
        $display("---------------------------------------------------------------");
        
        in = 4'b0000; #10;
        $display("No bits set: in=%04b, pos=%02b (expect 00)", in, pos);
        check_result(2'b00, "Zero input");
        
        // Test 5: Dynamic changes
        $display("\nTest 5: Dynamic Priority Changes");
        $display("---------------------------------------------------------------");
        
        in = 4'b0001; #10;
        $display("Start with bit 0: pos=%02b", pos);
        
        in = 4'b0011; #10;
        $display("Add bit 1: pos=%02b (should change to 01)", pos);
        
        in = 4'b0111; #10;
        $display("Add bit 2: pos=%02b (should change to 10)", pos);
        
        in = 4'b1111; #10;
        $display("Add bit 3: pos=%02b (should change to 11)", pos);
        
        in = 4'b1110; #10;
        $display("Remove bit 0: pos=%02b (should stay 11)", pos);
        
        in = 4'b1000; #10;
        $display("Only bit 3: pos=%02b (should stay 11)", pos);
        
        // Test 6: Casez functionality verification
        $display("\nTest 6: Casez Pattern Matching");
        $display("---------------------------------------------------------------");
        $display("Verifying casez with don't care (?) patterns:");
        
        in = 4'b1001; #10;
        $display("Pattern 1???: in=%04b matches, pos=%02b", in, pos);
        
        in = 4'b0110; #10;
        $display("Pattern 01??: in=%04b matches, pos=%02b", in, pos);
        
        in = 4'b0010; #10;
        $display("Pattern 001?: in=%04b matches, pos=%02b", in, pos);
        
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
    
    // Task to check result
    task check_result;
        input [1:0] expected;
        input [20*8:1] description;
        begin
            total_tests = total_tests + 1;
            
            if (pos == expected) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  Result: PASS (%0s)", description);
            end else begin
                $display("  Result: FAIL (%0s) - got %02b, expected %02b", 
                    description, pos, expected);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("priority_encoder_tb.vcd");
        $dumpvars(0, tb_top_module_always_case);
    end
    
endmodule