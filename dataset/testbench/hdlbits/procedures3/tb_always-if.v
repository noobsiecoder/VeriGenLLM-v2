`timescale 1ns/1ps

module tb_top_module_always();
    // Inputs
    reg a;
    reg b;
    reg sel_b1;
    reg sel_b2;
    
    // Outputs
    wire out_assign;
    wire out_always;
    
    // Instantiate DUT
    top_module_always dut (
        .a(a),
        .b(b),
        .sel_b1(sel_b1),
        .sel_b2(sel_b2),
        .out_assign(out_assign),
        .out_always(out_always)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg expected_out;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 2-to-1 Multiplexer with AND condition");
        $display("Select b when BOTH sel_b1 AND sel_b2 are true");
        $display("Otherwise select a");
        $display("===============================================================");
        
        // Test 1: Complete truth table
        $display("\nTest 1: Complete Truth Table");
        $display("---------------------------------------------------------------");
        $display("sel_b1 | sel_b2 | a | b | assign | always | Expected | Result");
        $display("---------------------------------------------------------------");
        
        // Test all 16 combinations
        for (i = 0; i < 16; i = i + 1) begin
            {sel_b1, sel_b2, a, b} = i[3:0];
            
            // Calculate expected output
            if (sel_b1 && sel_b2)
                expected_out = b;
            else
                expected_out = a;
            
            #10;
            
            total_tests = total_tests + 1;
            
            if (out_assign == expected_out && out_always == expected_out && 
                out_assign == out_always) begin
                num_tests_passed = num_tests_passed + 1;
                $display("   %b   |   %b    | %b | %b |   %b    |   %b    |    %b     | PASS",
                    sel_b1, sel_b2, a, b, out_assign, out_always, expected_out);
            end else begin
                $display("   %b   |   %b    | %b | %b |   %b    |   %b    |    %b     | FAIL",
                    sel_b1, sel_b2, a, b, out_assign, out_always, expected_out);
            end
        end
        
        // Test 2: Specific select conditions
        $display("\nTest 2: Select Condition Analysis");
        $display("---------------------------------------------------------------");
        
        // Case 1: Both selects false - should select a
        a = 1; b = 0;
        sel_b1 = 0; sel_b2 = 0;
        #10;
        $display("sel_b1=0, sel_b2=0: Output=%b (should be a=%b)", out_assign, a);
        check_result(a, "Both selects false");
        
        // Case 2: Only sel_b1 true - should select a
        sel_b1 = 1; sel_b2 = 0;
        #10;
        $display("sel_b1=1, sel_b2=0: Output=%b (should be a=%b)", out_assign, a);
        check_result(a, "Only sel_b1 true");
        
        // Case 3: Only sel_b2 true - should select a
        sel_b1 = 0; sel_b2 = 1;
        #10;
        $display("sel_b1=0, sel_b2=1: Output=%b (should be a=%b)", out_assign, a);
        check_result(a, "Only sel_b2 true");
        
        // Case 4: Both selects true - should select b
        sel_b1 = 1; sel_b2 = 1;
        #10;
        $display("sel_b1=1, sel_b2=1: Output=%b (should be b=%b)", out_assign, b);
        check_result(b, "Both selects true");
        
        // Test 3: Dynamic switching
        $display("\nTest 3: Dynamic Input Switching");
        $display("---------------------------------------------------------------");
        
        // Set different values for a and b
        a = 0; b = 1;
        $display("Setting a=0, b=1 for clear distinction");
        
        // Toggle through select combinations
        for (i = 0; i < 4; i = i + 1) begin
            {sel_b1, sel_b2} = i[1:0];
            #10;
            
            expected_out = (sel_b1 && sel_b2) ? b : a;
            
            $display("sel_b1=%b, sel_b2=%b: out=%b (expected %b) - Selecting %s",
                sel_b1, sel_b2, out_assign, expected_out,
                (sel_b1 && sel_b2) ? "b" : "a");
        end
        
        // Test 4: Timing comparison
        $display("\nTest 4: Implementation Timing Comparison");
        $display("---------------------------------------------------------------");
        
        // Set initial state
        a = 1; b = 0;
        sel_b1 = 0; sel_b2 = 0;
        #10;
        
        // Change both selects simultaneously
        $display("Changing both selects 0->1 simultaneously:");
        sel_b1 = 1; sel_b2 = 1;
        #1;
        
        $display("After 1ns: assign=%b, always=%b", out_assign, out_always);
        
        if (out_assign == out_always && out_assign == b) begin
            $display("Both implementations switched simultaneously: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Implementations differ or incorrect: FAIL");
        end
        total_tests = total_tests + 1;
        
        // Test 5: Glitch test
        $display("\nTest 5: Glitch Sensitivity Test");
        $display("---------------------------------------------------------------");
        
        a = 0; b = 1;
        
        // Start with b selected
        sel_b1 = 1; sel_b2 = 1;
        #10;
        
        // Create potential glitch by toggling sel_b1
        $display("Toggling sel_b1 while sel_b2=1:");
        sel_b1 = 0; #1;  // Should switch to a
        $display("  sel_b1=0: out=%b (should be a=%b)", out_assign, a);
        
        sel_b1 = 1; #1;  // Should switch back to b
        $display("  sel_b1=1: out=%b (should be b=%b)", out_assign, b);
        
        // Test 6: Random patterns
        $display("\nTest 6: Random Test Patterns");
        $display("---------------------------------------------------------------");
        
        repeat(10) begin
            a = $random;
            b = $random;
            sel_b1 = $random;
            sel_b2 = $random;
            
            #10;
            
            expected_out = (sel_b1 && sel_b2) ? b : a;
            
            total_tests = total_tests + 1;
            if (out_assign == expected_out && out_always == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
                $display("Random: sel=(%b,%b), a=%b, b=%b, out=%b - PASS",
                    sel_b1, sel_b2, a, b, out_assign);
            end else begin
                $display("Random: sel=(%b,%b), a=%b, b=%b, out=%b, exp=%b - FAIL",
                    sel_b1, sel_b2, a, b, out_assign, expected_out);
            end
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
    
    // Task to check result
    task check_result;
        input expected;
        input [30*8:1] description;
        begin
            total_tests = total_tests + 1;
            
            if (out_assign == expected && out_always == expected) begin
                num_tests_passed = num_tests_passed + 1;
                $display("Result: PASS (%0s)", description);
            end else begin
                $display("Result: FAIL (%0s) - assign=%b, always=%b, expected=%b",
                    description, out_assign, out_always, expected);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("mux_2to1_and_tb.vcd");
        $dumpvars(0, tb_top_module_always);
    end
    
endmodule