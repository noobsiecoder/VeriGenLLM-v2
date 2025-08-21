`timescale 1ns/1ps

module tb_four_wires();
    // Inputs (driven by testbench)
    reg a, b, c;
    
    // Outputs (driven by DUT)
    wire w, x, y, z;
    
    // Instantiate the Design Under Test (DUT)
    four_wires dut (
        .a(a), .b(b), .c(c),
        .w(w), .x(x), .y(y), .z(z)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("=========================================");
        $display("Testing four_wires module");
        $display("=========================================");
        $display("Expected connections: a->w, b->x, b->y, c->z");
        $display("-----------------------------------------");
        $display("Time  | a b c | w x y z | Result");
        $display("-----------------------------------------");
        
        // Test all 8 possible input combinations
        for (i = 0; i < 8; i = i + 1) begin
            {a, b, c} = i[2:0];  // Assign 3-bit value to inputs
            #10;  // Wait for propagation
            
            // Check all connections
            if ((w == a) && (x == b) && (y == b) && (z == c)) begin
                $display("%3t   | %b %b %b | %b %b %b %b | PASS", 
                         $time, a, b, c, w, x, y, z);
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("%3t   | %b %b %b | %b %b %b %b | FAIL", 
                         $time, a, b, c, w, x, y, z);
                
                // Show what's wrong
                if (w !== a) $display("  ERROR: w=%b but should be %b (a)", w, a);
                if (x !== b) $display("  ERROR: x=%b but should be %b (b)", x, b);
                if (y !== b) $display("  ERROR: y=%b but should be %b (b)", y, b);
                if (z !== c) $display("  ERROR: z=%b but should be %b (c)", z, c);
            end
            total_tests = total_tests + 1;
        end
        
        // Additional dynamic test
        $display("\nDynamic random test:");
        repeat(10) begin
            a = $random;
            b = $random;
            c = $random;
            #5;
            
            if ((w == a) && (x == b) && (y == b) && (z == c)) begin
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("FAIL at time %0t: inputs=%b%b%b, outputs=%b%b%b%b", 
                         $time, a, b, c, w, x, y, z);
            end
            total_tests = total_tests + 1;
        end
        
        // Final Summary
        $display("\n=========================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else
            $display("Overall Result: SOME TESTS FAILED ✗");
        $display("=========================================");
        
        $finish;
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("four_wires_tb.vcd");
        $dumpvars(0, tb_four_wires);
    end
    
endmodule