`timescale 1ns / 1ps

module tb_simple_wire_assign;
    
    // Output from module under test
    wire a;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    simple_wire_assign_example uut(.a(a));
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 2: Assign a simple wire with a high value");
        $display("Description: Set a wire to logic high using assign");
        $display("Expected: Output 'a' should always be 1'b1");
        $display("====================================================\n");
        
        // Test 1: Check initial value
        #10;
        test_count = test_count + 1;
        $display("Test 1: Checking initial output value...");
        if (a === 1'b1) begin
            $display("  PASS: Output 'a' is high (1'b1)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Output 'a' is %b, expected 1'b1", a);
            test_passed = 1'b0;
        end
        
        // Additional check: Verify no width mismatch warnings during compilation
        // This would need to be checked externally as Verilog can't detect its own warnings
        
        // Test 2: Check value remains stable over time
        #50;
        test_count = test_count + 1;
        $display("\nTest 2: Checking output stability after 50ns...");
        if (a === 1'b1) begin
            $display("  PASS: Output 'a' remains high (1'b1)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Output 'a' is %b, expected 1'b1", a);
            test_passed = 1'b0;
        end
        
        // Test 3: Check value one more time
        #100;
        test_count = test_count + 1;
        $display("\nTest 3: Final check after 150ns total...");
        if (a === 1'b1) begin
            $display("  PASS: Output 'a' still high (1'b1)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Output 'a' is %b, expected 1'b1", a);
            test_passed = 1'b0;
        end
        
        // Display test summary
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
    
    // Timeout to prevent hanging
    initial begin
        #1000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule