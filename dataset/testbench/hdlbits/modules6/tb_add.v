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

// Testbench for top_module_add
module tb_top_module_add();
    // Inputs
    reg [31:0] a;
    reg [31:0] b;
    
    // Output
    wire [31:0] sum;
    
    // Instantiate DUT
    top_module_add dut (
        .a(a),
        .b(b),
        .sum(sum)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [31:0] expected_sum;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        a = 32'h0;
        b = 32'h0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 32-bit Adder using two 16-bit Adders");
        $display("Module structure: Two add16 modules with carry chain");
        $display("===============================================================");
        
        // Test 1: Basic addition tests
        $display("\nTest 1: Basic Addition Tests");
        $display("------------------------------------------------------------");
        $display("         a        |        b        |       sum       | Expected | Result");
        $display("------------------------------------------------------------");
        
        // Test zero addition
        test_addition(32'h00000000, 32'h00000000, "0 + 0");
        
        // Test simple additions
        test_addition(32'h00000001, 32'h00000001, "1 + 1");
        test_addition(32'h0000000F, 32'h00000001, "15 + 1");
        test_addition(32'h000000FF, 32'h00000001, "255 + 1");
        test_addition(32'h12345678, 32'h87654321, "Mixed");
        
        // Test 2: Carry propagation from lower to upper 16 bits
        $display("\nTest 2: Carry Propagation Tests");
        $display("------------------------------------------------------------");
        
        // Test carry from bit 15 to bit 16
        test_addition(32'h0000FFFF, 32'h00000001, "Carry to upper");
        test_addition(32'h0000FFFF, 32'h0000FFFF, "Max lower half");
        test_addition(32'h00008000, 32'h00008000, "Bit 15 carry");
        
        // Test 3: Upper 16-bit operations
        $display("\nTest 3: Upper 16-bit Operations");
        $display("------------------------------------------------------------");
        
        test_addition(32'h10000000, 32'h20000000, "Upper only");
        test_addition(32'hFFFF0000, 32'h00010000, "Upper carry");
        test_addition(32'hFFFF0000, 32'hFFFF0000, "Upper overflow");
        
        // Test 4: Maximum values
        $display("\nTest 4: Boundary Value Tests");
        $display("------------------------------------------------------------");
        
        test_addition(32'hFFFFFFFF, 32'h00000000, "Max + 0");
        test_addition(32'hFFFFFFFF, 32'h00000001, "Max + 1 (overflow)");
        test_addition(32'hFFFFFFFF, 32'hFFFFFFFF, "Max + Max");
        test_addition(32'h7FFFFFFF, 32'h7FFFFFFF, "Max positive");
        
        // Test 5: Random tests
        $display("\nTest 5: Random Addition Tests");
        $display("------------------------------------------------------------");
        
        for (i = 0; i < 10; i = i + 1) begin
            a = $random;
            b = $random;
            expected_sum = a + b;
            #10;
            
            total_tests = total_tests + 1;
            if (sum == expected_sum) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%08h | %08h | %08h | %08h | PASS", 
                    a, b, sum, expected_sum);
            end else begin
                $display("%08h | %08h | %08h | %08h | FAIL", 
                    a, b, sum, expected_sum);
            end
        end
        
        // Test 6: Carry chain verification
        $display("\nTest 6: Carry Chain Verification");
        $display("------------------------------------------------------------");
        
        // Set inputs that will generate carry from lower adder
        a = 32'h0000FFFF;
        b = 32'h00000001;
        #10;
        
        $display("Inputs: a=%08h, b=%08h", a, b);
        $display("Lower 16 bits: %04h + %04h = %04h", 
            a[15:0], b[15:0], sum[15:0]);
        $display("Carry from lower adder: cout1 = %b", dut.cout1);
        $display("Upper 16 bits: %04h + %04h + %b = %04h", 
            a[31:16], b[31:16], dut.cout1, sum[31:16]);
        $display("Final sum: %08h", sum);
        
        // Test 7: Detailed bit analysis
        $display("\nTest 7: Bit-level Analysis");
        $display("------------------------------------------------------------");
        
        a = 32'h5555AAAA;
        b = 32'hAAAA5555;
        #10;
        
        $display("a = %032b", a);
        $display("b = %032b", b);
        $display("sum = %032b", sum);
        $display("Expected = %032b", a + b);
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else if (num_tests_passed != 1'b0)
            $display("Overall Result: SOME TESTS PASSED ⚠");
        else
            $display("Overall Result: NO TESTS PASSED ✗");
        $display("===============================================================");
        
        $finish;
    end
    
    // Task to test addition
    task test_addition;
        input [31:0] test_a;
        input [31:0] test_b;
        input [15*8:1] description;
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
        $dumpfile("add32_tb.vcd");
        $dumpvars(0, tb_top_module_add);
    end
    
endmodule