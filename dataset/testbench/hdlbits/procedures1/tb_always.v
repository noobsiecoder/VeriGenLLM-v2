`timescale 1ns/1ps

module tb_top_module_always();
    // Inputs
    reg a;
    reg b;
    
    // Outputs
    wire out_assign;
    wire out_alwaysblock;
    
    // Instantiate DUT
    top_module_always dut (
        .a(a),
        .b(b),
        .out_assign(out_assign),
        .out_alwaysblock(out_alwaysblock)
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
        $display("Testing AND Gate: assign vs always block implementation");
        $display("Both outputs should always match for combinational logic");
        $display("===============================================================");
        
        // Test 1: Truth table verification
        $display("\nTest 1: AND Gate Truth Table");
        $display("-----------------------------------------------");
        $display("a | b | out_assign | out_always | Expected | Result");
        $display("-----------------------------------------------");
        
        // Test all 4 combinations
        for (i = 0; i < 4; i = i + 1) begin
            {a, b} = i[1:0];
            expected_out = a & b;
            #10;
            
            total_tests = total_tests + 1;
            
            if (out_assign == expected_out && out_alwaysblock == expected_out && 
                out_assign == out_alwaysblock) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%b | %b |     %b      |     %b      |    %b     | PASS",
                    a, b, out_assign, out_alwaysblock, expected_out);
            end else begin
                $display("%b | %b |     %b      |     %b      |    %b     | FAIL",
                    a, b, out_assign, out_alwaysblock, expected_out);
            end
        end
        
        // Test 2: Timing comparison
        $display("\nTest 2: Timing Comparison");
        $display("-----------------------------------------------");
        $display("Both outputs should change simultaneously");
        
        a = 0; b = 0;
        #10;
        
        // Change a from 0 to 1
        $display("\nChanging a: 0 -> 1 (b = 0)");
        a = 1;
        #1;
        $display("After 1ns: out_assign = %b, out_alwaysblock = %b", 
            out_assign, out_alwaysblock);
        if (out_assign == out_alwaysblock) begin
            $display("Outputs match: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Outputs don't match: FAIL");
        end
        total_tests = total_tests + 1;
        
        #9;
        
        // Change b from 0 to 1
        $display("\nChanging b: 0 -> 1 (a = 1)");
        b = 1;
        #1;
        $display("After 1ns: out_assign = %b, out_alwaysblock = %b", 
            out_assign, out_alwaysblock);
        if (out_assign == out_alwaysblock && out_assign == 1'b1) begin
            $display("Both outputs = 1: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Incorrect output: FAIL");
        end
        total_tests = total_tests + 1;
        
        // Test 3: Rapid transitions
        $display("\nTest 3: Rapid Input Transitions");
        $display("-----------------------------------------------");
        
        repeat(10) begin
            #5;
            a = $random;
            b = $random;
            #1; // Small delay to let signals settle
            
            expected_out = a & b;
            
            if (out_assign == expected_out && out_alwaysblock == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
            
            $display("a=%b, b=%b: assign=%b, always=%b, expected=%b %s",
                a, b, out_assign, out_alwaysblock, expected_out,
                (out_assign == expected_out && out_alwaysblock == expected_out) ? "PASS" : "FAIL");
        end
        
        // Test 4: Sensitivity verification
        $display("\nTest 4: Sensitivity List Verification");
        $display("-----------------------------------------------");
        $display("always @(*) should be sensitive to both a and b");
        
        a = 0; b = 1;
        #10;
        $display("Initial: a=%b, b=%b, out_always=%b", a, b, out_alwaysblock);
        
        a = 1; // Change only a
        #1;
        $display("After a change: out_always=%b (should be %b)", 
            out_alwaysblock, a & b);
        
        a = 0; b = 0;
        #10;
        b = 1; // Change only b
        #1;
        $display("After b change: out_always=%b (should be %b)", 
            out_alwaysblock, a & b);
        
        // Test 5: Implementation comparison
        $display("\nTest 5: Implementation Method Comparison");
        $display("-----------------------------------------------");
        $display("assign statement: Continuous assignment");
        $display("  - Always active, updates whenever inputs change");
        $display("  - Used for simple combinational logic");
        $display("  - Cannot be used for sequential logic");
        $display("\nalways @(*) block: Procedural assignment");
        $display("  - Executes when any signal in RHS changes");
        $display("  - Can contain more complex procedural code");
        $display("  - Output must be declared as 'reg'");
        
        // Test 6: Glitch detection
        $display("\nTest 6: Checking for Glitches");
        $display("-----------------------------------------------");
        
        a = 1; b = 1;
        #10;
        
        // Create a potential glitch scenario
        fork
            #0 a = 0;
            #0 b = 0;
        join
        
        #1;
        $display("Simultaneous change: a=%b, b=%b", a, b);
        $display("Outputs: assign=%b, always=%b", out_assign, out_alwaysblock);
        
        if (out_assign == 0 && out_alwaysblock == 0) begin
            $display("No glitch detected: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Possible glitch: FAIL");
        end
        total_tests = total_tests + 1;
        
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
    
    // Monitor to track changes
    initial begin
        $monitor("Time=%0t: a=%b, b=%b, out_assign=%b, out_alwaysblock=%b", 
            $time, a, b, out_assign, out_alwaysblock);
    end
    
    // Generate VCD file
    initial begin
        $dumpfile("and_gate_tb.vcd");
        $dumpvars(0, tb_top_module_always);
    end
    
endmodule