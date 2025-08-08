`timescale 1ns / 1ps

module tb_permute_8bit;
    
    // Input
    reg [7:0] in;
    
    // Output
    wire [7:0] out;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    
    // Expected permutation mapping:
    // out[0] = in[3]
    // out[1] = in[7]
    // out[2] = in[1]
    // out[3] = in[4]
    // out[4] = in[0]
    // out[5] = in[6]
    // out[6] = in[2]
    // out[7] = in[5]
    
    // Instantiate the module under test
    permute_8bit uut(
        .in(in),
        .out(out)
    );
    
    // Function to calculate expected permutation
    function [7:0] expected_permutation;
        input [7:0] input_val;
        begin
            expected_permutation[0] = input_val[3];
            expected_permutation[1] = input_val[7];
            expected_permutation[2] = input_val[1];
            expected_permutation[3] = input_val[4];
            expected_permutation[4] = input_val[0];
            expected_permutation[5] = input_val[6];
            expected_permutation[6] = input_val[2];
            expected_permutation[7] = input_val[5];
        end
    endfunction
    
    // Task to check permutation
    task check_permutation;
        input [7:0] test_input;
        input [127:0] test_description;
        reg [7:0] expected;
        begin
            in = test_input;
            #10; // Wait for combinational logic
            expected = expected_permutation(test_input);
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Input:    %b (%h)", in, in);
            $display("  Expected: %b (%h)", expected, expected);
            $display("  Got:      %b (%h)", out, out);
            
            if (out === expected) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
                // Show bit mapping
                $display("  Bit mapping check:");
                for (i = 0; i < 8; i = i + 1) begin
                    case (i)
                        0: if (out[0] !== in[3]) $display("    out[0] != in[3]");
                        1: if (out[1] !== in[7]) $display("    out[1] != in[7]");
                        2: if (out[2] !== in[1]) $display("    out[2] != in[1]");
                        3: if (out[3] !== in[4]) $display("    out[3] != in[4]");
                        4: if (out[4] !== in[0]) $display("    out[4] != in[0]");
                        5: if (out[5] !== in[6]) $display("    out[5] != in[6]");
                        6: if (out[6] !== in[2]) $display("    out[6] != in[2]");
                        7: if (out[7] !== in[5]) $display("    out[7] != in[5]");
                    endcase
                end
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 12: Bit Permutation");
        $display("Description: Reorder bits according to fixed mapping");
        $display("Permutation mapping:");
        $display("  out[0] <- in[3]");
        $display("  out[1] <- in[7]");
        $display("  out[2] <- in[1]");
        $display("  out[3] <- in[4]");
        $display("  out[4] <- in[0]");
        $display("  out[5] <- in[6]");
        $display("  out[6] <- in[2]");
        $display("  out[7] <- in[5]");
        $display("====================================================\n");
        
        // Test 1: All zeros
        check_permutation(8'b00000000, "All zeros");
        
        // Test 2: All ones
        check_permutation(8'b11111111, "All ones");
        
        // Test 3: Single bit tests (test each input bit separately)
        $display("Testing individual bit mappings:");
        check_permutation(8'b00000001, "Only in[0] set -> out[4]");
        check_permutation(8'b00000010, "Only in[1] set -> out[2]");
        check_permutation(8'b00000100, "Only in[2] set -> out[6]");
        check_permutation(8'b00001000, "Only in[3] set -> out[0]");
        check_permutation(8'b00010000, "Only in[4] set -> out[3]");
        check_permutation(8'b00100000, "Only in[5] set -> out[7]");
        check_permutation(8'b01000000, "Only in[6] set -> out[5]");
        check_permutation(8'b10000000, "Only in[7] set -> out[1]");
        
        // Test 4: Alternating patterns
        check_permutation(8'b10101010, "Alternating pattern 10101010");
        check_permutation(8'b01010101, "Alternating pattern 01010101");
        
        // Test 5: Sequential patterns
        check_permutation(8'b00001111, "Lower nibble set");
        check_permutation(8'b11110000, "Upper nibble set");
        
        // Test 6: Random patterns
        check_permutation(8'b11001100, "Pattern 11001100");
        check_permutation(8'b00110011, "Pattern 00110011");
        check_permutation(8'b10110101, "Random pattern 10110101");
        
        // Test 7: Verify it's purely combinational
        $display("Testing combinational behavior:");
        in = 8'b11100111;
        #5;
        if (out === expected_permutation(8'b11100111)) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output updates immediately", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Output not combinational", test_count);
            test_passed = 1'b0;
        end
        
        // Change input and verify immediate update
        in = 8'b00011000;
        #5;
        if (out === expected_permutation(8'b00011000)) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Output changes with input", test_count);
        end else begin
            test_count = test_count + 1;
            $display("Test %0d: FAIL - Output doesn't follow input", test_count);
            test_passed = 1'b0;
        end
        
        // Test 8: Verify the inverse would work (conceptual test)
        $display("\nVerifying permutation is consistent:");
        in = 8'b10011011;
        #10;
        test_count = test_count + 1;
        $display("Test %0d: Permutation consistency check", test_count);
        $display("  Input:  %b", in);
        $display("  Output: %b", out);
        $display("  Mapping verified bit by bit");
        
        // Check each bit mapping explicitly
        if (out[0] === in[3] && out[1] === in[7] && out[2] === in[1] && 
            out[3] === in[4] && out[4] === in[0] && out[5] === in[6] && 
            out[6] === in[2] && out[7] === in[5]) begin
            $display("  PASS - All bit mappings correct");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL - Bit mapping error");
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