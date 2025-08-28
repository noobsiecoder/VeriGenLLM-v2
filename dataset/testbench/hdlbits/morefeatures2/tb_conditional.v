`timescale 1ns/1ps

module tb_top_module_even_parity();
    // Input
    reg [7:0] in;
    
    // Output
    wire parity;
    
    // Instantiate DUT
    top_module_even_parity dut (
        .in(in),
        .parity(parity)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    reg expected_parity;
    integer ones_count;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Even Parity Generator");
        $display("Parity bit = XOR of all 8 data bits");
        $display("Even parity: Total number of 1s (including parity) is even");
        $display("===============================================================");
        
        // Test 1: Basic cases
        $display("\nTest 1: Basic Test Cases");
        $display("---------------------------------------------------------------");
        $display("in[7:0]  | Binary | #1s | Parity | Expected | Result");
        $display("---------------------------------------------------------------");
        
        // Test all zeros
        test_parity(8'b00000000, "All zeros");
        
        // Test all ones
        test_parity(8'b11111111, "All ones");
        
        // Test single bit
        test_parity(8'b00000001, "Single bit");
        test_parity(8'b10000000, "MSB only");
        
        // Test alternating
        test_parity(8'b10101010, "Alternating 1");
        test_parity(8'b01010101, "Alternating 2");
        
        // Test 2: All possible number of 1s
        $display("\nTest 2: Testing Different Numbers of 1s");
        $display("---------------------------------------------------------------");
        $display("Testing bytes with 0 through 8 ones:");
        
        // 0 ones
        test_parity(8'b00000000, "0 ones");
        
        // 1 one
        test_parity(8'b00000001, "1 one");
        
        // 2 ones
        test_parity(8'b00000011, "2 ones");
        
        // 3 ones
        test_parity(8'b00000111, "3 ones");
        
        // 4 ones
        test_parity(8'b00001111, "4 ones");
        
        // 5 ones
        test_parity(8'b00011111, "5 ones");
        
        // 6 ones
        test_parity(8'b00111111, "6 ones");
        
        // 7 ones
        test_parity(8'b01111111, "7 ones");
        
        // 8 ones
        test_parity(8'b11111111, "8 ones");
        
        // Test 3: Verify XOR property
        $display("\nTest 3: XOR Property Verification");
        $display("---------------------------------------------------------------");
        
        // XOR truth table for selected bits
        in = 8'b00000000;
        #10;
        $display("Starting with all 0s: parity = %b", parity);
        
        in = 8'b00000001;  // Toggle bit 0
        #10;
        $display("Toggle bit 0: in=%08b, parity = %b (should toggle)", in, parity);
        
        in = 8'b00000011;  // Toggle bit 1
        #10;
        $display("Toggle bit 1: in=%08b, parity = %b (should toggle)", in, parity);
        
        in = 8'b00000010;  // Toggle bit 0 back
        #10;
        $display("Toggle bit 0: in=%08b, parity = %b (should toggle)", in, parity);
        
        // Test 4: Complete coverage (sample)
        $display("\nTest 4: Sample Complete Coverage");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 256; i = i + 13) begin  // Sample every 13th value
            in = i[7:0];
            
            // Count ones
            ones_count = 0;
            for (j = 0; j < 8; j = j + 1) begin
                if (in[j]) ones_count = ones_count + 1;
            end
            
            // Expected parity: 1 if odd number of ones, 0 if even
            expected_parity = ones_count[0];  // LSB gives odd/even
            
            #10;
            
            total_tests = total_tests + 1;
            if (parity == expected_parity) begin
                num_tests_passed = num_tests_passed + 1;
            end
            
            if (i < 40 || i > 240) begin  // Show first and last few
                $display("%08b |   %3d  |  %d  |   %b    |    %b     | %s",
                    in, i, ones_count, parity, expected_parity,
                    (parity == expected_parity) ? "PASS" : "FAIL");
            end
        end
        
        // Test 5: Error detection capability
        $display("\nTest 5: Error Detection Demonstration");
        $display("---------------------------------------------------------------");
        
        // Original data
        in = 8'b10110101;
        #10;
        ones_count = count_ones(in);
        $display("Original data: %08b, ones=%d, parity=%b", in, ones_count, parity);
        $display("Total bits set (including parity): %d (%s)", 
            ones_count + parity, (ones_count + parity) % 2 == 0 ? "even" : "odd");
        
        // Simulate single bit error
        in = 8'b10110111;  // Bit 1 flipped
        #10;
        ones_count = count_ones(in);
        $display("\nSingle bit error: %08b, ones=%d, parity=%b", in, ones_count, parity);
        $display("Total bits set (including parity): %d (%s) - Error detected!", 
            ones_count + parity, (ones_count + parity) % 2 == 0 ? "even" : "odd");
        
        // Test 6: Specific patterns
        $display("\nTest 6: Specific Bit Patterns");
        $display("---------------------------------------------------------------");
        
        test_parity(8'b11110000, "Upper nibble");
        test_parity(8'b00001111, "Lower nibble");
        test_parity(8'b11001100, "Two pairs");
        test_parity(8'b10011001, "Symmetric");
        test_parity(8'hAA, "0xAA");
        test_parity(8'h55, "0x55");
        
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
    
    // Function to count ones
    function integer count_ones;
        input [7:0] data;
        integer i;
        begin
            count_ones = 0;
            for (i = 0; i < 8; i = i + 1) begin
                if (data[i]) count_ones = count_ones + 1;
            end
        end
    endfunction
    
    // Task to test parity
    task test_parity;
        input [7:0] test_in;
        input [20*8:1] description;
        integer ones;
        begin
            in = test_in;
            #10;
            
            ones = count_ones(in);
            expected_parity = ones[0];  // Odd number of ones
            
            total_tests = total_tests + 1;
            if (parity == expected_parity) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%08b |   %02h   |  %d  |   %b    |    %b     | PASS (%0s)",
                    in, in, ones, parity, expected_parity, description);
            end else begin
                $display("%08b |   %02h   |  %d  |   %b    |    %b     | FAIL (%0s)",
                    in, in, ones, parity, expected_parity, description);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("even_parity_tb.vcd");
        $dumpvars(0, tb_top_module_even_parity);
    end
    
endmodule