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

// Testbench for adder-subtractor
module tb_top_module_addsub();
    // Inputs
    reg [31:0] a;
    reg [31:0] b;
    reg sub;
    
    // Output
    wire [31:0] sum;
    
    // Instantiate DUT
    top_module_addsub dut (
        .a(a),
        .b(b),
        .sub(sub),
        .sum(sum)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [31:0] expected_result;
    reg signed [31:0] signed_a, signed_b, signed_result;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 32-bit Adder-Subtractor");
        $display("sub=0: sum = a + b");
        $display("sub=1: sum = a - b (implemented as a + ~b + 1)");
        $display("===============================================================");
        
        // Test 1: Basic addition tests (sub = 0)
        $display("\nTest 1: Addition Tests (sub = 0)");
        $display("------------------------------------------------------------");
        $display("sub |        a        |        b        |       sum       | Expected | Result");
        $display("------------------------------------------------------------");
        
        sub = 0;
        test_operation(32'h00000000, 32'h00000000, "0 + 0");
        test_operation(32'h00000001, 32'h00000001, "1 + 1");
        test_operation(32'h000000FF, 32'h00000001, "255 + 1");
        test_operation(32'h12345678, 32'h87654321, "Mixed addition");
        test_operation(32'hFFFFFFFF, 32'h00000001, "Max + 1");
        test_operation(32'h0000FFFF, 32'h00000001, "Carry test");
        
        // Test 2: Basic subtraction tests (sub = 1)
        $display("\nTest 2: Subtraction Tests (sub = 1)");
        $display("------------------------------------------------------------");
        $display("sub |        a        |        b        |       sum       | Expected | Result");
        $display("------------------------------------------------------------");
        
        sub = 1;
        test_operation(32'h00000005, 32'h00000003, "5 - 3");
        test_operation(32'h00000003, 32'h00000005, "3 - 5");
        test_operation(32'h00000100, 32'h00000001, "256 - 1");
        test_operation(32'h00000000, 32'h00000001, "0 - 1");
        test_operation(32'hFFFFFFFF, 32'hFFFFFFFF, "-1 - (-1)");
        test_operation(32'h12345678, 32'h12345678, "a - a");
        
        // Test 3: XOR gate verification
        $display("\nTest 3: XOR Gate Verification");
        $display("------------------------------------------------------------");
        
        b = 32'h5A5A5A5A;
        sub = 0;
        #10;
        $display("sub=0: b = %08h, xor_sig = %08h (should equal b)", 
            b, dut.xor_sig);
        
        sub = 1;
        #10;
        $display("sub=1: b = %08h, xor_sig = %08h (should equal ~b = %08h)", 
            b, dut.xor_sig, ~b);
        
        if (dut.xor_sig == ~b) begin
            $display("XOR gate working correctly: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("XOR gate not working: FAIL");
        end
        total_tests = total_tests + 1;
        
        // Test 4: Carry propagation in subtraction
        $display("\nTest 4: Carry Propagation in Subtraction");
        $display("------------------------------------------------------------");
        
        sub = 1;
        a = 32'h00010000;  // Borrow will propagate from lower to upper half
        b = 32'h00000001;
        #10;
        expected_result = a - b;
        $display("a = %08h, b = %08h", a, b);
        $display("Expected: a - b = %08h", expected_result);
        $display("Actual: sum = %08h", sum);
        $display("Lower carry: cout1 = %b", dut.cout1);
        check_result(expected_result, "Borrow propagation");
        
        // Test 5: Signed arithmetic tests
        $display("\nTest 5: Signed Arithmetic Tests");
        $display("------------------------------------------------------------");
        
        // Positive + Positive
        sub = 0;
        test_signed_operation(32'h00000064, 32'h00000032, "100 + 50");
        
        // Positive - Positive
        sub = 1;
        test_signed_operation(32'h00000064, 32'h00000032, "100 - 50");
        
        // Positive - Negative (becomes addition)
        test_signed_operation(32'h00000064, 32'hFFFFFFCE, "100 - (-50)");
        
        // Negative - Positive
        test_signed_operation(32'hFFFFFF9C, 32'h00000032, "-100 - 50");
        
        // Test 6: Boundary cases
        $display("\nTest 6: Boundary Cases");
        $display("------------------------------------------------------------");
        
        // Maximum positive - 1
        sub = 1;
        test_operation(32'h7FFFFFFF, 32'h00000001, "MaxPos - 1");
        
        // Minimum negative - (-1)
        test_operation(32'h80000000, 32'hFFFFFFFF, "MinNeg - (-1)");
        
        // Overflow cases
        sub = 0;
        test_operation(32'h7FFFFFFF, 32'h00000001, "MaxPos + 1 (overflow)");
        
        // Test 7: Random tests
        $display("\nTest 7: Random Tests");
        $display("------------------------------------------------------------");
        
        for (i = 0; i < 10; i = i + 1) begin
            a = $random;
            b = $random;
            sub = $random & 1;
            
            if (sub == 0) begin
                test_operation(a, b, "Random add");
            end else begin
                test_operation(a, b, "Random sub");
            end
        end
        
        // Test 8: Implementation verification
        $display("\nTest 8: Implementation Verification");
        $display("------------------------------------------------------------");
        
        a = 32'h0000000A;  // 10
        b = 32'h00000003;  // 3
        
        sub = 1;
        #10;
        $display("Subtraction as addition:");
        $display("  a = %d", a);
        $display("  b = %d", b);
        $display("  ~b = %08h", ~b);
        $display("  ~b + 1 = %08h (two's complement of b)", ~b + 1);
        $display("  a + (~b + 1) = %d (should equal a - b = %d)", sum, a - b);
        
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
    
    // Task to test operation
    task test_operation;
        input [31:0] test_a;
        input [31:0] test_b;
        input [20*8:1] description;
        begin
            a = test_a;
            b = test_b;
            
            if (sub == 0)
                expected_result = test_a + test_b;
            else
                expected_result = test_a - test_b;
            
            #10;
            
            total_tests = total_tests + 1;
            
            if (sum == expected_result) begin
                num_tests_passed = num_tests_passed + 1;
                $display(" %b  | %08h | %08h | %08h | %08h | PASS (%0s)", 
                    sub, a, b, sum, expected_result, description);
            end else begin
                $display(" %b  | %08h | %08h | %08h | %08h | FAIL (%0s)", 
                    sub, a, b, sum, expected_result, description);
            end
        end
    endtask
    
    // Task to test signed operations
    task test_signed_operation;
        input [31:0] test_a;
        input [31:0] test_b;
        input [20*8:1] description;
        begin
            a = test_a;
            b = test_b;
            signed_a = test_a;
            signed_b = test_b;
            
            if (sub == 0) begin
                expected_result = test_a + test_b;
                signed_result = signed_a + signed_b;
            end else begin
                expected_result = test_a - test_b;
                signed_result = signed_a - signed_b;
            end
            
            #10;
            
            total_tests = total_tests + 1;
            
            $display("%0s: %0d %s %0d = %0d", 
                description, signed_a, sub ? "-" : "+", signed_b, $signed(sum));
            
            if (sum == expected_result) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  Binary: %08h %s %08h = %08h PASS", 
                    a, sub ? "-" : "+", b, sum);
            end else begin
                $display("  Binary: %08h %s %08h = %08h (expected %08h) FAIL", 
                    a, sub ? "-" : "+", b, sum, expected_result);
            end
        end
    endtask
    
    // Task to check result
    task check_result;
        input [31:0] expected;
        input [20*8:1] description;
        begin
            total_tests = total_tests + 1;
            
            if (sum == expected) begin
                num_tests_passed = num_tests_passed + 1;
                $display("Result: PASS (%0s)", description);
            end else begin
                $display("Result: FAIL (%0s)", description);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("addsub_tb.vcd");
        $dumpvars(0, tb_top_module_addsub);
    end
    
endmodule