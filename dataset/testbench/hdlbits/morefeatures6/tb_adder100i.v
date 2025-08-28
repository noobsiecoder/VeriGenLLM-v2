`timescale 1ns/1ps

module tb_top_module_adder100i();
    // Inputs
    reg [99:0] a, b;
    reg cin;
    
    // Outputs
    wire [99:0] cout;
    wire [99:0] sum;
    
    // Instantiate DUT
    top_module_adder100i dut (
        .a(a),
        .b(b),
        .cin(cin),
        .cout(cout),
        .sum(sum)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    reg [100:0] expected_sum;  // 101 bits to catch overflow
    reg [99:0] expected_cout;
    reg carry;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 100-bit Ripple-Carry Adder");
        $display("Verifying sum and all intermediate carry outputs");
        $display("===============================================================");
        
        // Test 1: Basic cases
        $display("\nTest 1: Basic Test Cases");
        $display("---------------------------------------------------------------");
        
        // Zero + Zero + 0
        a = 100'h0;
        b = 100'h0;
        cin = 1'b0;
        #10;
        expected_sum = 101'h0;
        check_result("0 + 0 + 0", expected_sum[99:0], expected_sum[100]);
        
        // Zero + Zero + 1
        a = 100'h0;
        b = 100'h0;
        cin = 1'b1;
        #10;
        expected_sum = 101'h1;
        check_result("0 + 0 + 1", expected_sum[99:0], expected_sum[100]);
        
        // All ones + zero + 0
        a = {100{1'b1}};
        b = 100'h0;
        cin = 1'b0;
        #10;
        expected_sum = {1'b0, {100{1'b1}}};
        check_result("All_1s + 0 + 0", expected_sum[99:0], expected_sum[100]);
        
        // All ones + zero + 1 (should overflow)
        a = {100{1'b1}};
        b = 100'h0;
        cin = 1'b1;
        #10;
        expected_sum = {1'b1, 100'h0};  // Overflow
        check_result("All_1s + 0 + 1", expected_sum[99:0], expected_sum[100]);
        
        // Test 2: Carry propagation
        $display("\nTest 2: Carry Propagation Tests");
        $display("---------------------------------------------------------------");
        
        // All ones + 1 (maximum carry propagation)
        a = {100{1'b1}};
        b = 100'h1;
        cin = 1'b0;
        #10;
        expected_sum = {1'b1, 100'h0};
        check_result("All_1s + 1", expected_sum[99:0], expected_sum[100]);
        verify_carry_chain("Full propagation");
        
        // Test partial carry propagation
        a = 100'hFF;  // Lower 8 bits set
        b = 100'h1;
        cin = 1'b0;
        #10;
        expected_sum = 101'h100;
        check_result("0xFF + 1", expected_sum[99:0], expected_sum[100]);
        
        // Test 3: Bit position tests
        $display("\nTest 3: Bit Position Tests");
        $display("---------------------------------------------------------------");
        
        // Test each bit position
        for (i = 0; i < 10; i = i + 1) begin
            a = 100'h1 << (i * 10);
            b = 100'h1 << (i * 10);
            cin = 1'b0;
            #10;
            
            expected_sum = 101'h0;
            expected_sum[i * 10 + 1] = 1'b1;
            
            $display("Bit %2d: sum[%2d]=%b, cout[%2d]=%b %s", 
                i * 10, i * 10 + 1, sum[i * 10 + 1], i * 10, cout[i * 10],
                (sum == expected_sum[99:0] && cout[i * 10] == 1'b1) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (sum == expected_sum[99:0] && cout[i * 10] == 1'b1) 
                num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 4: Random addition tests
        $display("\nTest 4: Random Addition Tests");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 20; i = i + 1) begin
            // Generate random inputs
            a[31:0] = $random;
            a[63:32] = $random;
            a[95:64] = $random;
            a[99:96] = $random & 4'hF;
            
            b[31:0] = $random;
            b[63:32] = $random;
            b[95:64] = $random;
            b[99:96] = $random & 4'hF;
            
            cin = $random & 1'b1;
            
            #10;
            
            // Calculate expected result
            expected_sum = a + b + cin;
            
            $display("Random %2d: %s", i,
                (sum == expected_sum[99:0] && cout[99] == expected_sum[100]) ? "PASS" : "FAIL");
                
            total_tests = total_tests + 1;
            if (sum == expected_sum[99:0] && cout[99] == expected_sum[100])
                num_tests_passed = num_tests_passed + 1;
        end
        
        // Test 5: Carry chain verification
        $display("\nTest 5: Detailed Carry Chain Tests");
        $display("---------------------------------------------------------------");
        
        // Create a pattern that generates specific carry patterns
        // Note: 0x555... has 25 hex digits for 100 bits (25*4 = 100)
        a = 100'h5555555555555555555555555;  // Alternating 0101...
        b = 100'hAAAAAAAAAAAAAAAAAAAAAAAAA;  // Alternating 1010...
        cin = 1'b1;
        #10;
        
        // 0101... + 1010... = 1111... (all ones)
        // 1111... + 1 = 0000... with carry out
        expected_sum = {1'b1, 100'h0};
        check_result("Alternating patterns", expected_sum[99:0], expected_sum[100]);
        
        // Verify carry propagation through all bits
        carry = cin;
        for (i = 0; i < 100; i = i + 1) begin
            expected_cout[i] = (a[i] & b[i]) | (a[i] & carry) | (b[i] & carry);
            carry = expected_cout[i];
        end
        
        if (cout == expected_cout) begin
            $display("Carry chain verification: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Carry chain verification: FAIL");
            // Find first mismatch
            i = 0;
            while (i < 100 && cout[i] == expected_cout[i]) begin
                i = i + 1;
            end
            if (i < 100) begin
                $display("  First carry mismatch at bit %d: got %b, expected %b",
                    i, cout[i], expected_cout[i]);
            end
        end
        total_tests = total_tests + 1;
        
        // Test 6: Edge cases
        $display("\nTest 6: Edge Cases");
        $display("---------------------------------------------------------------");
        
        // Maximum value - 1 + 1 + 1
        a = {100{1'b1}} - 1;
        b = 100'h1;
        cin = 1'b1;
        #10;
        expected_sum = {1'b1, 100'h0};
        check_result("Max-1 + 1 + 1", expected_sum[99:0], expected_sum[100]);
        
        // Half adder behavior (cin = 0)
        a = 100'h123456789ABCDEF0123456789;
        b = 100'h9876543210FEDCBA9876543;
        cin = 1'b0;
        #10;
        expected_sum = a + b;
        check_result("Complex pattern, cin=0", expected_sum[99:0], expected_sum[100]);
        
        // Same pattern with cin = 1
        cin = 1'b1;
        #10;
        expected_sum = a + b + 1;
        check_result("Complex pattern, cin=1", expected_sum[99:0], expected_sum[100]);
        
        // Final Summary
        $display("\n===============================================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, total_tests);
        if (num_tests_passed == total_tests)
            $display("Overall Result: ALL TESTS PASSED ✓");
        else
            $display("Overall Result: SOME TESTS FAILED ⚠");
        $display("===============================================================");
        
        $finish;
    end
    
    // Task to check result
    task check_result;
        input [50*8:1] description;
        input [99:0] expected_sum;
        input expected_carry;
        begin
            total_tests = total_tests + 1;
            
            if (sum == expected_sum && cout[99] == expected_carry) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%0s: PASS", description);
            end else begin
                $display("%0s: FAIL", description);
                $display("  Expected: sum=%h, cout[99]=%b", expected_sum, expected_carry);
                $display("  Got:      sum=%h, cout[99]=%b", sum, cout[99]);
            end
        end
    endtask
    
    // Task to verify carry chain for all 1s input
    task verify_carry_chain;
        input [50*8:1] description;
        integer k;
        reg all_carries_one;
        begin
            all_carries_one = 1'b1;
            for (k = 0; k < 100; k = k + 1) begin
                if (cout[k] != 1'b1) all_carries_one = 1'b0;
            end
            
            $display("%0s carry chain: %s", description,
                all_carries_one ? "All carries = 1 PASS" : "FAIL");
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("adder100i_tb.vcd");
        $dumpvars(0, tb_top_module_adder100i);
    end
    
endmodule