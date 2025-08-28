`timescale 1ns/1ps

// Define add16 module for testing
module add16 (
    input [15:0] a,
    input [15:0] b,
    input cin,
    output [15:0] sum,
    output cout
);
    assign {cout, sum} = a + b + cin;
endmodule

// Testbench for carry-select adder
module tb_top_module_cseladd();
    // Inputs
    reg [31:0] a;
    reg [31:0] b;
    
    // Output
    wire [31:0] sum;
    
    // Instantiate DUT
    top_module_cseladd dut (
        .a(a),
        .b(b),
        .sum(sum)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [31:0] expected_sum;
    
    // Timing measurement
    time start_time, end_time;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Carry-Select Adder (32-bit)");
        $display("Architecture: Parallel computation with multiplexed selection");
        $display("===============================================================");
        
        // Wait for initial settling
        #10;
        
        // Test 1: Basic functionality tests
        $display("\nTest 1: Basic Addition Tests");
        $display("------------------------------------------------------------");
        $display("         a        |        b        |       sum       | Expected | Result");
        $display("------------------------------------------------------------");
        
        test_addition(32'h00000000, 32'h00000000, "0 + 0");
        test_addition(32'h00000001, 32'h00000001, "1 + 1");
        test_addition(32'h000000FF, 32'h00000001, "255 + 1");
        test_addition(32'h12345678, 32'h87654321, "Mixed");
        test_addition(32'hFFFFFFFF, 32'h00000001, "Max + 1");
        
        // Test 2: Carry propagation verification
        $display("\nTest 2: Carry Propagation Tests");
        $display("------------------------------------------------------------");
        
        // Test carry from lower to upper half
        test_addition(32'h0000FFFF, 32'h00000001, "Carry to upper");
        test_addition(32'h0000FFFF, 32'h0000FFFF, "Double carry");
        test_addition(32'h00008000, 32'h00008000, "Bit 15 carry");
        test_addition(32'h0000FFFE, 32'h00000003, "Near boundary");
        
        // Test 3: Multiplexer selection verification
        $display("\nTest 3: Multiplexer Selection Verification");
        $display("------------------------------------------------------------");
        
        // Case where cout = 0 (no carry from lower half)
        a = 32'h00001234;
        b = 32'h00005678;
        #10;
        $display("No carry case:");
        $display("  Inputs: a=%08h, b=%08h", a, b);
        $display("  Lower: %04h + %04h = %04h, cout = %b", 
            a[15:0], b[15:0], sum[15:0], dut.cout);
        $display("  Upper (cin=0): %04h + %04h = %04h", 
            a[31:16], b[31:16], dut.sum1);
        $display("  Upper (cin=1): %04h + %04h + 1 = %04h", 
            a[31:16], b[31:16], dut.sum2);
        $display("  Selected: sum[31:16] = %04h (should use cin=0 result)", sum[31:16]);
        $display("  Verification: %s", (sum[31:16] == dut.sum1) ? "PASS" : "FAIL");
        
        // Case where cout = 1 (carry from lower half)
        a = 32'h0000FFFF;
        b = 32'h00000001;
        #10;
        $display("\nCarry case:");
        $display("  Inputs: a=%08h, b=%08h", a, b);
        $display("  Lower: %04h + %04h = %04h, cout = %b", 
            a[15:0], b[15:0], sum[15:0], dut.cout);
        $display("  Upper (cin=0): %04h + %04h = %04h", 
            a[31:16], b[31:16], dut.sum1);
        $display("  Upper (cin=1): %04h + %04h + 1 = %04h", 
            a[31:16], b[31:16], dut.sum2);
        $display("  Selected: sum[31:16] = %04h (should use cin=1 result)", sum[31:16]);
        $display("  Verification: %s", (sum[31:16] == dut.sum2) ? "PASS" : "FAIL");
        
        // Test 4: Edge cases
        $display("\nTest 4: Edge Cases");
        $display("------------------------------------------------------------");
        
        // Maximum values
        test_addition(32'hFFFFFFFF, 32'hFFFFFFFF, "Max + Max");
        test_addition(32'h7FFFFFFF, 32'h7FFFFFFF, "Max signed");
        test_addition(32'h80000000, 32'h80000000, "Min signed");
        
        // Power of 2 boundaries
        test_addition(32'h00010000, 32'hFFFF0000, "16-bit boundary");
        test_addition(32'h01000000, 32'h01000000, "Large values");
        
        // Test 5: Random comprehensive tests
        $display("\nTest 5: Random Addition Tests");
        $display("------------------------------------------------------------");
        
        for (i = 0; i < 20; i = i + 1) begin
            a = $random;
            b = $random;
            test_addition(a, b, "Random");
        end
        
        // Test 6: Specific multiplexer switching tests
        $display("\nTest 6: Multiplexer Switching Tests");
        $display("------------------------------------------------------------");
        
        // Test cases right at the carry boundary
        test_mux_operation(32'h0000FFFE, 32'h00000001, "No carry - boundary");
        test_mux_operation(32'h0000FFFE, 32'h00000002, "Carry - boundary");
        test_mux_operation(32'h0000FFFF, 32'h00000000, "No carry - max lower");
        test_mux_operation(32'h0000FFFF, 32'h00000001, "Carry - max lower");
        
        // Test 7: Architecture verification
        $display("\nTest 7: Architecture Verification");
        $display("------------------------------------------------------------");
        $display("Carry-Select Adder Structure:");
        $display("  1. Lower 16-bit adder (always cin=0)");
        $display("  2. Upper 16-bit adder with cin=0 (speculative)");
        $display("  3. Upper 16-bit adder with cin=1 (speculative)");
        $display("  4. Multiplexer selects based on lower carry-out");
        $display("\nTiming advantage:");
        $display("  - Both upper additions compute in parallel");
        $display("  - Only MUX delay added after lower carry-out");
        $display("  - Faster than waiting for carry to ripple through");
        
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
    
    // Task to test addition
    task test_addition;
        input [31:0] test_a;
        input [31:0] test_b;
        input [20*8:1] description;
        begin
            a = test_a;
            b = test_b;
            expected_sum = test_a + test_b;
            #10;
            
            total_tests = total_tests + 1;
            
            if (sum == expected_sum) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%08h | %08h | %08h | %08h | PASS (%0s)", 
                    a, b, sum, expected_sum, description);
            end else begin
                $display("%08h | %08h | %08h | %08h | FAIL (%0s)", 
                    a, b, sum, expected_sum, description);
            end
        end
    endtask
    
    // Task to specifically test multiplexer operation
    task test_mux_operation;
        input [31:0] test_a;
        input [31:0] test_b;
        input [20*8:1] description;
        begin
            a = test_a;
            b = test_b;
            expected_sum = test_a + test_b;
            #10;
            
            total_tests = total_tests + 1;
            
            $display("\n%0s:", description);
            $display("  a=%08h, b=%08h", a, b);
            $display("  Lower carry: cout = %b", dut.cout);
            $display("  MUX selected: %s path", dut.cout ? "cin=1" : "cin=0");
            $display("  Result: %08h, Expected: %08h", sum, expected_sum);
            
            if (sum == expected_sum) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  Status: PASS");
            end else begin
                $display("  Status: FAIL");
            end
        end
    endtask
    
    // Monitor for debugging
    initial begin
        $monitor("Time=%0t: a=%h, b=%h, sum=%h, cout=%b, sum1=%h, sum2=%h", 
            $time, a, b, sum, dut.cout, dut.sum1, dut.sum2);
    end
    
    // Generate VCD file
    initial begin
        $dumpfile("cseladd_tb.vcd");
        $dumpvars(0, tb_top_module_cseladd);
    end
    
endmodule