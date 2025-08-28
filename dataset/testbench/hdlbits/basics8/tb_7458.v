`timescale 1ns/1ps

module tb_microcontroller_7458();
    // Inputs (driven by testbench)
    reg p1a, p1b, p1c, p1d, p1e, p1f;
    reg p2a, p2b, p2c, p2d;
    
    // Outputs (driven by DUT)
    wire p1y, p2y;
    
    // Instantiate the Design Under Test (DUT)
    microcontroller_7458 dut (
        .p1a(p1a), .p1b(p1b), .p1c(p1c), 
        .p1d(p1d), .p1e(p1e), .p1f(p1f),
        .p1y(p1y),
        .p2a(p2a), .p2b(p2b), .p2c(p2c), .p2d(p2d),
        .p2y(p2y)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    
    // Expected values
    reg expected_p1y;
    reg expected_p2y;
    
    // Intermediate signals for clarity
    reg W, X, Y, Z;
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing microcontroller_7458 module");
        $display("Function: p1y = (p1a & p1c & p1b) | (p1f & p1e & p1d)");
        $display("          p2y = (p2a & p2b) | (p2c & p2d)");
        $display("===============================================================");
        
        // Test Case 1: All zeros
        $display("\nTest Case 1: All inputs = 0");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b000000;
        {p2a, p2b, p2c, p2d} = 4'b0000;
        #10;
        expected_p1y = 1'b0;
        expected_p2y = 1'b0;
        check_outputs("All zeros", expected_p1y, expected_p2y);
        
        // Test Case 2: Test first AND gate (W)
        $display("\nTest Case 2: Testing W gate (p1a & p1c & p1b)");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b111000;
        {p2a, p2b, p2c, p2d} = 4'b0000;
        #10;
        expected_p1y = 1'b1;  // W = 1, Y = 0, so p1y = 1
        expected_p2y = 1'b0;  // X = 0, Z = 0, so p2y = 0
        check_outputs("W=1, others=0", expected_p1y, expected_p2y);
        
        // Test Case 3: Test second AND gate (X)
        $display("\nTest Case 3: Testing X gate (p2a & p2b)");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b000000;
        {p2a, p2b, p2c, p2d} = 4'b1100;
        #10;
        expected_p1y = 1'b0;
        expected_p2y = 1'b1;  // X = 1, Z = 0, so p2y = 1
        check_outputs("X=1, others=0", expected_p1y, expected_p2y);
        
        // Test Case 4: Test third AND gate (Y)
        $display("\nTest Case 4: Testing Y gate (p1f & p1e & p1d)");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b000111;
        {p2a, p2b, p2c, p2d} = 4'b0000;
        #10;
        expected_p1y = 1'b1;  // W = 0, Y = 1, so p1y = 1
        expected_p2y = 1'b0;
        check_outputs("Y=1, others=0", expected_p1y, expected_p2y);
        
        // Test Case 5: Test fourth AND gate (Z)
        $display("\nTest Case 5: Testing Z gate (p2c & p2d)");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b000000;
        {p2a, p2b, p2c, p2d} = 4'b0011;
        #10;
        expected_p1y = 1'b0;
        expected_p2y = 1'b1;  // X = 0, Z = 1, so p2y = 1
        check_outputs("Z=1, others=0", expected_p1y, expected_p2y);
        
        // Test Case 6: Both W and Y active
        $display("\nTest Case 6: Both W and Y active");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b111111;
        {p2a, p2b, p2c, p2d} = 4'b0000;
        #10;
        expected_p1y = 1'b1;  // W = 1, Y = 1, so p1y = 1
        expected_p2y = 1'b0;
        check_outputs("W=1, Y=1", expected_p1y, expected_p2y);
        
        // Test Case 7: Both X and Z active
        $display("\nTest Case 7: Both X and Z active");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b000000;
        {p2a, p2b, p2c, p2d} = 4'b1111;
        #10;
        expected_p1y = 1'b0;
        expected_p2y = 1'b1;  // X = 1, Z = 1, so p2y = 1
        check_outputs("X=1, Z=1", expected_p1y, expected_p2y);
        
        // Test Case 8: All gates active
        $display("\nTest Case 8: All AND gates active");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b111111;
        {p2a, p2b, p2c, p2d} = 4'b1111;
        #10;
        expected_p1y = 1'b1;
        expected_p2y = 1'b1;
        check_outputs("All AND gates=1", expected_p1y, expected_p2y);
        
        // Test Case 9: Partial inputs (testing that all inputs must be 1 for AND)
        $display("\nTest Case 9: Partial inputs for W gate");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b110000;  // p1c = 0
        {p2a, p2b, p2c, p2d} = 4'b0000;
        #10;
        expected_p1y = 1'b0;  // W = 0 (because p1c = 0)
        expected_p2y = 1'b0;
        check_outputs("W partial (p1c=0)", expected_p1y, expected_p2y);
        
        // Test Case 10: Random pattern
        $display("\nTest Case 10: Random pattern");
        {p1a, p1b, p1c, p1d, p1e, p1f} = 6'b101011;
        {p2a, p2b, p2c, p2d} = 4'b1010;
        #10;
        W = p1a & p1c & p1b;  // 1 & 1 & 0 = 0
        Y = p1f & p1e & p1d;  // 1 & 1 & 0 = 0
        X = p2a & p2b;        // 1 & 0 = 0
        Z = p2c & p2d;        // 1 & 0 = 0
        expected_p1y = W | Y;  // 0 | 0 = 0
        expected_p2y = X | Z;  // 0 | 0 = 0
        check_outputs("Random pattern", expected_p1y, expected_p2y);
        
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
    
    // Task to check outputs and display results
    task check_outputs;
        input [50*8:1] test_name;
        input expected_p1y_val;
        input expected_p2y_val;
        begin
            total_tests = total_tests + 1;
            
            // Calculate intermediate values for display
            W = p1a & p1c & p1b;
            X = p2a & p2b;
            Y = p1f & p1e & p1d;
            Z = p2c & p2d;
            
            $display("Inputs: p1[a,b,c,d,e,f]=%b%b%b%b%b%b, p2[a,b,c,d]=%b%b%b%b", 
                p1a, p1b, p1c, p1d, p1e, p1f, p2a, p2b, p2c, p2d);
            $display("Gates:  W=%b, X=%b, Y=%b, Z=%b", W, X, Y, Z);
            $display("Expected: p1y=%b, p2y=%b", expected_p1y_val, expected_p2y_val);
            $display("Actual:   p1y=%b, p2y=%b", p1y, p2y);
            
            if (p1y == expected_p1y_val && p2y == expected_p2y_val) begin
                $display("Result: PASS");
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("Result: FAIL");
            end
        end
    endtask
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("microcontroller_7458_tb.vcd");
        $dumpvars(0, tb_microcontroller_7458);
    end
    
endmodule