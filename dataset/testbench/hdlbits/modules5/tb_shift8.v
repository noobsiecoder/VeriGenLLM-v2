`timescale 1ns/1ps

// Define my_dff8 module for testing
module my_dff8 (
    input wire clk,
    input wire [7:0] d,
    output reg [7:0] q
);
    initial begin
        q = 8'b0;
    end
    
    always @(posedge clk) begin
        q <= d;
    end
endmodule

// Testbench for top_module_shift
module tb_top_module_shift();
    // Inputs
    reg clk;
    reg [7:0] d;
    reg [1:0] sel;
    
    // Output
    wire [7:0] q;
    
    // Instantiate DUT
    top_module_shift dut (
        .clk(clk),
        .d(d),
        .sel(sel),
        .q(q)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i, j;
    reg [7:0] expected_q;
    reg [7:0] expected_reg [0:2];
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        d = 8'h00;
        sel = 2'b00;
        
        // Display header
        $display("===============================================================");
        $display("Testing 8-bit Shift Register with Output Multiplexer");
        $display("sel=0: q=d (bypass), sel=1: q=sig_a (1 delay)");
        $display("sel=2: q=sig_b (2 delays), sel=3: q=sig_c (3 delays)");
        $display("===============================================================");
        
        // Wait for initialization
        #10;
        @(posedge clk);
        
        // Test 1: Verify multiplexer selection
        $display("\nTest 1: Multiplexer Selection Test");
        $display("----------------------------------------------------------");
        $display("Time | d    | sel | sig_a | sig_b | sig_c | q    | Expected | Result");
        $display("---------------------------------------------------------------------");
        
        // Load a pattern
        d = 8'hAA;
        @(posedge clk); #1;
        d = 8'hBB;
        @(posedge clk); #1;
        d = 8'hCC;
        @(posedge clk); #1;
        d = 8'hDD;
        
        // Test each selector
        for (i = 0; i < 4; i = i + 1) begin
            sel = i[1:0];
            #1;
            case (sel)
                2'b00: expected_q = d;
                2'b01: expected_q = dut.sig_a;
                2'b10: expected_q = dut.sig_b;
                2'b11: expected_q = dut.sig_c;
            endcase
            
            check_result(sel, expected_q);
        end
        
        // Test 2: Shift register propagation
        $display("\nTest 2: Shift Register Propagation");
        $display("----------------------------------------------------------");
        $display("Cycle | d    | sig_a | sig_b | sig_c | q (sel=3)");
        $display("----------------------------------------------------------");
        
        sel = 2'b11; // Select sig_c output
        d = 8'h00;
        
        // Clear the pipeline
        repeat(3) @(posedge clk);
        
        // Test pattern
        for (i = 0; i < 8; i = i + 1) begin
            d = 8'h11 * (i + 1); // 11, 22, 33, etc.
            @(posedge clk); #1;
            $display("  %0d   | %02h   | %02h    | %02h    | %02h    | %02h",
                i, d, dut.sig_a, dut.sig_b, dut.sig_c, q);
        end
        
        // Test 3: Dynamic selector changes
        $display("\nTest 3: Dynamic Selector Changes");
        $display("----------------------------------------------------------");
        
        d = 8'h55;
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk); #1;
            for (j = 0; j < 4; j = j + 1) begin
                sel = j[1:0];
                #1;
                $display("Time=%0t: d=%02h, sel=%0d, q=%02h", 
                    $time, d, sel, q);
            end
        end
        
        // Test 4: Walking ones pattern
        $display("\nTest 4: Walking Ones Pattern");
        $display("----------------------------------------------------------");
        
        sel = 2'b11; // Watch 3-cycle delayed output
        for (i = 0; i < 8; i = i + 1) begin
            d = 8'h01 << i;
            @(posedge clk); #1;
            $display("Shift %0d: d=%08b, q=%08b", i, d, q);
        end
        
        // Test 5: Comprehensive verification
        $display("\nTest 5: Comprehensive Verification");
        $display("----------------------------------------------------------");
        
        // Test all selector values with known data
        d = 8'h00;
        repeat(3) @(posedge clk); // Clear pipeline
        
        // Load test values
        d = 8'hF1; @(posedge clk); #1;  // Goes to sig_a
        d = 8'hF2; @(posedge clk); #1;  // F1->sig_b, F2->sig_a
        d = 8'hF3; @(posedge clk); #1;  // F1->sig_c, F2->sig_b, F3->sig_a
        d = 8'hF4; #1;                   // Current input
        
        // Now test all selectors
        $display("Current state: d=F4, sig_a=F3, sig_b=F2, sig_c=F1");
        
        sel = 2'b00; #1;
        verify_output("sel=0 (bypass)", 8'hF4);
        
        sel = 2'b01; #1;
        verify_output("sel=1 (1 delay)", 8'hF3);
        
        sel = 2'b10; #1;
        verify_output("sel=2 (2 delays)", 8'hF2);
        
        sel = 2'b11; #1;
        verify_output("sel=3 (3 delays)", 8'hF1);
        
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
    
    // Task to check results with selector value
    task check_result;
        input [1:0] test_sel;
        input [7:0] exp_q;
        begin
            total_tests = total_tests + 1;
            
            $display("%3t  | %02h  | %0d   | %02h    | %02h    | %02h    | %02h  | %02h      | %s",
                $time, d, test_sel, dut.sig_a, dut.sig_b, dut.sig_c, q, exp_q,
                (q == exp_q) ? "PASS" : "FAIL");
            
            if (q == exp_q) begin
                num_tests_passed = num_tests_passed + 1;
            end
        end
    endtask
    
    // Task to verify output with description
    task verify_output;
        input [20*8:1] description;
        input [7:0] exp_q;
        begin
            total_tests = total_tests + 1;
            
            if (q == exp_q) begin
                num_tests_passed = num_tests_passed + 1;
                $display("%0s: q=%02h, expected=%02h - PASS", description, q, exp_q);
            end else begin
                $display("%0s: q=%02h, expected=%02h - FAIL", description, q, exp_q);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("shift_mux_tb.vcd");
        $dumpvars(0, tb_top_module_shift);
    end
    
endmodule