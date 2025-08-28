`timescale 1ns/1ps

module tb_vector_concat();
    // Inputs (driven by testbench)
    reg [4:0] a, b, c, d, e, f;
    
    // Outputs (driven by DUT)
    wire [7:0] w, x, y, z;
    
    // Instantiate the Design Under Test (DUT)
    vector_concat dut (
        .a(a), .b(b), .c(c), .d(d), .e(e), .f(f),
        .w(w), .x(x), .y(y), .z(z)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    
    // Expected values
    reg [31:0] expected_concat;
    reg [7:0] expected_w, expected_x, expected_y, expected_z;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Vector Concatenation Module");
        $display("Function: {a, b, c, d, e, f, 2'b11} => 32-bit split to 8-bit outputs");
        $display("Total bits: 5*6 + 2 = 32 bits");
        $display("===============================================================");
        
        // Test Case 1: All zeros (except the 2'b11)
        $display("\nTest Case 1: All zeros");
        a = 5'b00000; b = 5'b00000; c = 5'b00000;
        d = 5'b00000; e = 5'b00000; f = 5'b00000;
        expected_concat = {a, b, c, d, e, f, 2'b11};
        expected_w = expected_concat[31:24];
        expected_x = expected_concat[23:16];
        expected_y = expected_concat[15:8];
        expected_z = expected_concat[7:0];
        #10;
        display_and_check("All zeros", expected_w, expected_x, expected_y, expected_z);
        
        // Test Case 2: All ones
        $display("\nTest Case 2: All ones");
        a = 5'b11111; b = 5'b11111; c = 5'b11111;
        d = 5'b11111; e = 5'b11111; f = 5'b11111;
        expected_concat = {a, b, c, d, e, f, 2'b11};
        expected_w = expected_concat[31:24];
        expected_x = expected_concat[23:16];
        expected_y = expected_concat[15:8];
        expected_z = expected_concat[7:0];
        #10;
        display_and_check("All ones", expected_w, expected_x, expected_y, expected_z);
        
        // Test Case 3: Sequential pattern
        $display("\nTest Case 3: Sequential pattern");
        a = 5'b00001; b = 5'b00010; c = 5'b00011;
        d = 5'b00100; e = 5'b00101; f = 5'b00110;
        expected_concat = {a, b, c, d, e, f, 2'b11};
        expected_w = expected_concat[31:24];
        expected_x = expected_concat[23:16];
        expected_y = expected_concat[15:8];
        expected_z = expected_concat[7:0];
        #10;
        display_and_check("Sequential", expected_w, expected_x, expected_y, expected_z);
        
        // Test Case 4: Individual input visibility
        $display("\nTest Case 4: Testing individual input visibility");
        $display("Setting only one input at a time to 5'b11111:");
        
        // Only a is set
        a = 5'b11111; b = 5'b00000; c = 5'b00000;
        d = 5'b00000; e = 5'b00000; f = 5'b00000;
        #10;
        $display("  Only a=11111: concat=%032b", {a, b, c, d, e, f, 2'b11});
        $display("  w=%08b, x=%08b, y=%08b, z=%08b", w, x, y, z);
        
        // Only b is set
        a = 5'b00000; b = 5'b11111; c = 5'b00000;
        d = 5'b00000; e = 5'b00000; f = 5'b00000;
        #10;
        $display("  Only b=11111: concat=%032b", {a, b, c, d, e, f, 2'b11});
        $display("  w=%08b, x=%08b, y=%08b, z=%08b", w, x, y, z);
        
        // Test Case 5: Boundary check - verifying bit positions
        $display("\nTest Case 5: Bit position verification");
        a = 5'b10000; b = 5'b01000; c = 5'b00100;
        d = 5'b00010; e = 5'b00001; f = 5'b10001;
        expected_concat = {a, b, c, d, e, f, 2'b11};
        #10;
        $display("Concatenated: %032b", expected_concat);
        $display("Bit mapping:");
        $display("  Bits [31:27] = a = %05b", a);
        $display("  Bits [26:22] = b = %05b", b);
        $display("  Bits [21:17] = c = %05b", c);
        $display("  Bits [16:12] = d = %05b", d);
        $display("  Bits [11:7]  = e = %05b", e);
        $display("  Bits [6:2]   = f = %05b", f);
        $display("  Bits [1:0]   = 11");
        $display("Output mapping:");
        $display("  w[7:0] = bits[31:24] = %08b", w);
        $display("  x[7:0] = bits[23:16] = %08b", x);
        $display("  y[7:0] = bits[15:8]  = %08b", y);
        $display("  z[7:0] = bits[7:0]   = %08b", z);
        
        // Test Case 6: Random patterns
        $display("\nTest Case 6: Random patterns");
        
        a = 5'b10101; b = 5'b01010; c = 5'b11001;
        d = 5'b00110; e = 5'b11110; f = 5'b00011;
        expected_concat = {a, b, c, d, e, f, 2'b11};
        expected_w = expected_concat[31:24];
        expected_x = expected_concat[23:16];
        expected_y = expected_concat[15:8];
        expected_z = expected_concat[7:0];
        #10;
        display_and_check("Random 1", expected_w, expected_x, expected_y, expected_z);
        
        a = 5'b11100; b = 5'b00111; c = 5'b10000;
        d = 5'b01111; e = 5'b10010; f = 5'b01101;
        expected_concat = {a, b, c, d, e, f, 2'b11};
        expected_w = expected_concat[31:24];
        expected_x = expected_concat[23:16];
        expected_y = expected_concat[15:8];
        expected_z = expected_concat[7:0];
        #10;
        display_and_check("Random 2", expected_w, expected_x, expected_y, expected_z);
        
        // Test Case 7: Verify the constant 2'b11
        $display("\nTest Case 7: Verifying constant 2'b11 in LSBs");
        a = 5'b00000; b = 5'b00000; c = 5'b00000;
        d = 5'b00000; e = 5'b00000; f = 5'b00000;
        #10;
        $display("With all inputs zero, z[1:0] should be 11: z=%08b", z);
        $display("z[1:0] = %02b %s", z[1:0], (z[1:0] == 2'b11) ? "✓" : "✗");
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else if (num_tests_passed != 0)
            $display("Overall Result: SOME TESTS PASSED ⚠");
        else
            $display("Overall Result: NO TESTS PASSED ✗");
        $display("===============================================================");
        
        $finish;
    end
    
    // Task to display and check outputs
    task display_and_check;
        input [20*8:1] test_name;
        input [7:0] exp_w, exp_x, exp_y, exp_z;
        begin
            total_tests = total_tests + 1;
            
            $display("Inputs: a=%05b, b=%05b, c=%05b, d=%05b, e=%05b, f=%05b",
                a, b, c, d, e, f);
            $display("Expected: w=%08b, x=%08b, y=%08b, z=%08b",
                exp_w, exp_x, exp_y, exp_z);
            $display("Actual:   w=%08b, x=%08b, y=%08b, z=%08b",
                w, x, y, z);
            
            if (w == exp_w && x == exp_x && y == exp_y && z == exp_z) begin
                $display("Result: PASS");
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("Result: FAIL");
                if (w != exp_w) $display("  w mismatch!");
                if (x != exp_x) $display("  x mismatch!");
                if (y != exp_y) $display("  y mismatch!");
                if (z != exp_z) $display("  z mismatch!");
            end
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("vector_concat_tb.vcd");
        $dumpvars(0, tb_vector_concat);
    end
    
endmodule