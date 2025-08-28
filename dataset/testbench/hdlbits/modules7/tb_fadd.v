`timescale 1ns/1ps

// Provided add16 module (for testing purposes)
module add16 ( 
    input [15:0] a, 
    input [15:0] b, 
    input cin, 
    output [15:0] sum, 
    output cout 
);
    wire [16:0] c;  // Carry chain
    assign c[0] = cin;
    
    // Generate 16 full adders
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : adder_chain
            add1 fa(
                .a(a[i]),
                .b(b[i]),
                .cin(c[i]),
                .sum(sum[i]),
                .cout(c[i+1])
            );
        end
    endgenerate
    
    assign cout = c[16];
endmodule

// Testbench
module tb_top_module_fadd();
    // Inputs
    reg [31:0] a;
    reg [31:0] b;
    
    // Output
    wire [31:0] sum;
    
    // Instantiate DUT
    top_module_fadd dut (
        .a(a),
        .b(b),
        .sum(sum)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [31:0] expected_sum;
    
    // For testing individual add1 module - FIXED: Made these 1-bit
    reg test_a, test_b, test_cin;
    wire test_sum, test_cout;
    reg test_expected_sum;    // 1-bit
    reg test_expected_cout;   // 1-bit
    
    // Instantiate single add1 for unit testing
    add1 test_add1(
        .a(test_a),
        .b(test_b),
        .cin(test_cin),
        .sum(test_sum),
        .cout(test_cout)
    );
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Hierarchical 32-bit Adder");
        $display("Structure: top_module_fadd -> 2x add16 -> 32x add1");
        $display("===============================================================");
        
        // Test 1: Unit test add1 module
        $display("\nTest 1: Testing add1 Full Adder Module");
        $display("-----------------------------------------------");
        $display("a | b | cin | sum | cout | Expected | Result");
        $display("-----------------------------------------------");
        
        // Test all 8 combinations for full adder
        for (i = 0; i < 8; i = i + 1) begin
            {test_a, test_b, test_cin} = i[2:0];
            #10;
            
            // Calculate expected values for 1-bit full adder
            {test_expected_cout, test_expected_sum} = test_a + test_b + test_cin;
            
            total_tests = total_tests + 1;
            if (test_sum == test_expected_sum && test_cout == test_expected_cout) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%b | %b |  %b  |  %b  |  %b   | s=%b c=%b | PASS",
                    test_a, test_b, test_cin, test_sum, test_cout, 
                    test_expected_sum, test_expected_cout);
            end else begin
                $display("%b | %b |  %b  |  %b  |  %b   | s=%b c=%b | FAIL",
                    test_a, test_b, test_cin, test_sum, test_cout, 
                    test_expected_sum, test_expected_cout);
            end
        end
        
        // Test 2: Basic 32-bit addition
        $display("\nTest 2: Basic 32-bit Addition Tests");
        $display("------------------------------------------------------------");
        $display("         a        |        b        |       sum       | Expected | Result");
        $display("------------------------------------------------------------");
        
        // Test cases
        test_addition(32'h00000000, 32'h00000000, "0 + 0");
        test_addition(32'h00000001, 32'h00000001, "1 + 1");
        test_addition(32'h000000FF, 32'h00000001, "255 + 1");
        test_addition(32'h12345678, 32'h87654321, "Mixed");
        
        // Test 3: Carry propagation tests
        $display("\nTest 3: Carry Propagation Tests");
        $display("------------------------------------------------------------");
        
        // Test carry from bit to bit
        test_addition(32'h00000001, 32'h00000001, "Simple carry");
        test_addition(32'h0000000F, 32'h00000001, "Nibble carry");
        test_addition(32'h000000FF, 32'h00000001, "Byte carry");
        test_addition(32'h00000FFF, 32'h00000001, "12-bit carry");
        test_addition(32'h0000FFFF, 32'h00000001, "16-bit boundary");
        test_addition(32'h000FFFFF, 32'h00000001, "20-bit carry");
        test_addition(32'hFFFFFFFF, 32'h00000001, "Full carry chain");
        
        // Test 4: Random comprehensive tests
        $display("\nTest 4: Random Addition Tests");
        $display("------------------------------------------------------------");
        
        for (i = 0; i < 10; i = i + 1) begin
            a = $random;
            b = $random;
            test_addition(a, b, "Random");
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
    
    // Generate VCD file
    initial begin
        $dumpfile("hierarchical_add32_tb.vcd");
        $dumpvars(0, tb_top_module_fadd);
    end
    
endmodule