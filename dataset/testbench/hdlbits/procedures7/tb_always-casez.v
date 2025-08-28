`timescale 1ns/1ps

module tb_top_module_always_case();
    // Input
    reg [7:0] in;
    
    // Output
    wire [2:0] pos;
    
    // Instantiate DUT
    top_module_always_case dut (
        .in(in),
        .pos(pos)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    reg [2:0] expected_pos;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 8-bit Priority Encoder (LSB Priority)");
        $display("Finds position of FIRST (least significant) 1 bit");
        $display("Output: 0-7 for bit position, 0 if no bits set");
        $display("===============================================================");
        
        // Test 1: Single bit set cases
        $display("\nTest 1: Single Bit Set Cases");
        $display("---------------------------------------------------------------");
        $display("in[7:0]  | Binary | pos | Expected | Result");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 8; i = i + 1) begin
            in = 8'b1 << i;
            expected_pos = i[2:0];
            #10;
            
            total_tests = total_tests + 1;
            if (pos == expected_pos) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%08b |   %3d  | %03b |   %03b    | PASS", 
                    in, in, pos, expected_pos);
            end else begin
                $display("%08b |   %3d  | %03b |   %03b    | FAIL", 
                    in, in, pos, expected_pos);
            end
        end
        
        // Test 2: Zero input
        $display("\nTest 2: Zero Input Case");
        $display("---------------------------------------------------------------");
        
        in = 8'b00000000;
        expected_pos = 3'b000;
        #10;
        
        total_tests = total_tests + 1;
        if (pos == expected_pos) begin
            num_tests_passed = num_tests_passed + 1;
            $display("No bits set: in=%08b, pos=%03b - PASS", in, pos);
        end else begin
            $display("No bits set: in=%08b, pos=%03b (expected %03b) - FAIL", 
                in, pos, expected_pos);
        end
        
        // Test 3: Multiple bits set (LSB priority)
        $display("\nTest 3: Multiple Bits Set (LSB Priority Test)");
        $display("---------------------------------------------------------------");
        
        // All bits set
        in = 8'b11111111;
        expected_pos = 3'b000;  // Bit 0 is first
        #10;
        check_result("All bits set", expected_pos);
        
        // Even bits set
        in = 8'b10101010;
        expected_pos = 3'b001;  // Bit 1 is first
        #10;
        check_result("Even bits (10101010)", expected_pos);
        
        // Odd bits set
        in = 8'b01010101;
        expected_pos = 3'b000;  // Bit 0 is first
        #10;
        check_result("Odd bits (01010101)", expected_pos);
        
        // Upper nibble only
        in = 8'b11110000;
        expected_pos = 3'b100;  // Bit 4 is first
        #10;
        check_result("Upper nibble (11110000)", expected_pos);
        
        // Test 4: Specific test cases
        $display("\nTest 4: Specific Test Cases");
        $display("---------------------------------------------------------------");
        
        test_case(8'b10010000, 3'd4, "Example from problem");
        test_case(8'b00000010, 3'd1, "Only bit 1");
        test_case(8'b11111110, 3'd1, "All but bit 0");
        test_case(8'b10000000, 3'd7, "Only MSB");
        test_case(8'b00011000, 3'd3, "Bits 3 and 4");
        
        // Test 5: Systematic coverage
        $display("\nTest 5: Systematic Coverage (sample cases)");
        $display("---------------------------------------------------------------");
        
        // Test finding first 1 with various patterns
        for (i = 0; i < 8; i = i + 1) begin
            // Pattern: all zeros up to bit i, then all ones
            in = 8'hFF << i;
            expected_pos = (i == 8) ? 3'b000 : i[2:0];
            #10;
            
            $display("Pattern FF<<%0d: in=%08b, pos=%03b (expect %03b) %s",
                i, in, pos, expected_pos, 
                (pos == expected_pos) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (pos == expected_pos) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
        
        // Test 6: Random patterns
        $display("\nTest 6: Random Patterns");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 10; i = i + 1) begin
            in = $random & 8'hFF;
            
            // Calculate expected position
            expected_pos = 3'b000;
            for (j = 0; j < 8; j = j + 1) begin
                if (in[j] == 1'b1) begin
                    expected_pos = j[2:0];
                    j = 8;  // Exit loop
                end
            end
            
            #10;
            
            total_tests = total_tests + 1;
            if (pos == expected_pos) begin
                num_tests_passed = num_tests_passed + 1;
                $display("Random: in=%08b, pos=%03b - PASS", in, pos);
            end else begin
                $display("Random: in=%08b, pos=%03b (expected %03b) - FAIL", 
                    in, pos, expected_pos);
            end
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
    
    // Task to test specific case
    task test_case;
        input [7:0] test_in;
        input [2:0] expected;
        input [30*8:1] description;
        begin
            in = test_in;
            #10;
            
            total_tests = total_tests + 1;
            if (pos == expected) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%0s: in=%08b, pos=%03b - PASS", 
                    description, in, pos);
            end else begin
                $display("%0s: in=%08b, pos=%03b (expected %03b) - FAIL", 
                    description, in, pos, expected);
            end
        end
    endtask
    
    // Task to check result
    task check_result;
        input [30*8:1] description;
        input [2:0] expected;
        begin
            total_tests = total_tests + 1;
            
            if (pos == expected) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%0s: in=%08b, pos=%03b - PASS", 
                    description, in, pos);
            end else begin
                $display("%0s: in=%08b, pos=%03b (expected %03b) - FAIL", 
                    description, in, pos, expected);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("priority_encoder_lsb_tb.vcd");
        $dumpvars(0, tb_top_module_always_case);
    end
    
endmodule