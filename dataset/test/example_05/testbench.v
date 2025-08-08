`timescale 1ns / 1ps

module tb_mux2to1;
    
    // Inputs
    reg a, b, sel;
    
    // Output
    wire y;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    mux2to1 uut(
        .a(a),
        .b(b),
        .sel(sel),
        .y(y)
    );
    
    // Task to check multiplexer output
    task check_mux;
        input test_a, test_b, test_sel;
        input expected_y;
        input [127:0] test_description;
        begin
            a = test_a;
            b = test_b;
            sel = test_sel;
            #10; // Wait for propagation
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Inputs: a=%b, b=%b, sel=%b", a, b, sel);
            $display("  Expected: y=%b", expected_y);
            $display("  Got:      y=%b", y);
            
            if (y === expected_y) begin
                $display("  PASS");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL");
                test_passed = 1'b0;
            end
            $display("");
        end
    endtask
    
    initial begin
        $display("\n================== TESTBENCH START ==================");
        $display("Problem 5: 2-input Multiplexer");
        $display("Description: MUX selects between inputs a and b");
        $display("Function: y = sel ? b : a");
        $display("  sel=0: y=a");
        $display("  sel=1: y=b");
        $display("====================================================\n");
        
        // Test all possible input combinations
        $display("Running exhaustive tests (all 8 combinations):\n");
        
        // Test 1: sel=0, should select 'a'
        check_mux(1'b0, 1'b0, 1'b0, 1'b0, "sel=0, a=0, b=0 -> y=a=0");
        check_mux(1'b1, 1'b0, 1'b0, 1'b1, "sel=0, a=1, b=0 -> y=a=1");
        check_mux(1'b0, 1'b1, 1'b0, 1'b0, "sel=0, a=0, b=1 -> y=a=0");
        check_mux(1'b1, 1'b1, 1'b0, 1'b1, "sel=0, a=1, b=1 -> y=a=1");
        
        // Test 2: sel=1, should select 'b'
        check_mux(1'b0, 1'b0, 1'b1, 1'b0, "sel=1, a=0, b=0 -> y=b=0");
        check_mux(1'b1, 1'b0, 1'b1, 1'b0, "sel=1, a=1, b=0 -> y=b=0");
        check_mux(1'b0, 1'b1, 1'b1, 1'b1, "sel=1, a=0, b=1 -> y=b=1");
        check_mux(1'b1, 1'b1, 1'b1, 1'b1, "sel=1, a=1, b=1 -> y=b=1");
        
        // Dynamic test - changing select line
        $display("Dynamic test - switching select line:\n");
        a = 1'b0;
        b = 1'b1;
        sel = 1'b0;
        #10;
        test_count = test_count + 1;
        $display("Test %0d: Initial state", test_count);
        $display("  a=0, b=1, sel=0 -> y=%b (should be 0)", y);
        if (y === 1'b0) begin
            pass_count = pass_count + 1;
            $display("  PASS");
        end else begin
            test_passed = 1'b0;
            $display("  FAIL");
        end
        
        sel = 1'b1;  // Switch select
        #10;
        test_count = test_count + 1;
        $display("\nTest %0d: After switching sel", test_count);
        $display("  a=0, b=1, sel=1 -> y=%b (should be 1)", y);
        if (y === 1'b1) begin
            pass_count = pass_count + 1;
            $display("  PASS");
        end else begin
            test_passed = 1'b0;
            $display("  FAIL");
        end
        
        // Test with unknown/high-impedance values
        $display("\nEdge case tests:\n");
        
        // Test with unknown select
        test_count = test_count + 1;
        a = 1'b0;
        b = 1'b1;
        sel = 1'bx;
        #10;
        $display("Test %0d: Unknown select (sel=x)", test_count);
        $display("  a=0, b=1, sel=x -> y=%b", y);
        if (y === 1'bx) begin
            $display("  PASS: Unknown propagated correctly");
            pass_count = pass_count + 1;
        end else begin
            $display("  WARNING: Implementation specific behavior");
            // Don't fail on this edge case
        end
        
        // Display test summary
        $display("\n================== TEST RESULTS ====================");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", test_count - pass_count);
        
        if (test_passed && pass_count >= 10) begin  // At least the main tests must pass
            $display("\nRESULT: ALL TESTS PASSED ✓");
        end else begin
            $display("\nRESULT: TESTS FAILED ✗");
        end
        $display("====================================================\n");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #1000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule