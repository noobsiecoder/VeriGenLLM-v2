`timescale 1ns/1ps

// Define my_dff with reset capability for proper initialization
module my_dff (
    input wire clk,
    input wire d,
    output reg q
);
    // Initialize q to 0
    initial begin
        q = 1'b0;
    end
    
    // D flip-flop implementation
    always @(posedge clk) begin
        q <= d;
    end
endmodule

// Testbench for top_module_shift
module tb_top_module_shift();
    // Inputs (driven by testbench)
    reg clk;
    reg d;
    
    // Outputs (driven by DUT)
    wire q;
    
    // Instantiate the Design Under Test (DUT)
    top_module_shift dut (
        .clk(clk),
        .d(d),
        .q(q)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [2:0] expected_shift_reg;
    reg expected_q;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end
    
    initial begin
        // Initialize test tracking
        num_tests_passed = 0;
        total_tests = 0;
        expected_shift_reg = 3'b000;
        
        // Display header
        $display("===============================================================");
        $display("Testing top_module_shift - 3-stage shift register");
        $display("Module connections:");
        $display("  d -> flop1.d -> sig_a -> flop2.d -> sig_b -> flop3.d -> q");
        $display("  All flops share the same clock signal");
        $display("===============================================================");
        
        // Wait for initial settling
        d = 0;
        @(posedge clk);
        #1;
        
        // Test shifting pattern through the register
        $display("\nShift Register Operation:");
        $display("--------------------------------------------------------------");
        $display("Cycle | clk | d | sig_a | sig_b | q | Expected | Shift Reg | Result");
        $display("--------------------------------------------------------------");
        
        // Test pattern: 11011001
        test_shift(1'b1, 0);  // Shift in 1
        test_shift(1'b1, 1);  // Shift in 1
        test_shift(1'b0, 2);  // Shift in 0
        test_shift(1'b1, 3);  // Shift in 1
        test_shift(1'b1, 4);  // Shift in 1
        test_shift(1'b0, 5);  // Shift in 0
        test_shift(1'b0, 6);  // Shift in 0
        test_shift(1'b1, 7);  // Shift in 1
        
        // Flush the register
        test_shift(1'b0, 8);
        test_shift(1'b0, 9);
        test_shift(1'b0, 10);
        
        // Test all zeros
        $display("\n===============================================================");
        $display("All Zeros Test:");
        $display("--------------------------------------------------------------");
        
        for (i = 0; i < 5; i = i + 1) begin
            test_shift(1'b0, 11 + i);
        end
        
        // Test all ones
        $display("\n===============================================================");
        $display("All Ones Test:");
        $display("--------------------------------------------------------------");
        
        for (i = 0; i < 5; i = i + 1) begin
            test_shift(1'b1, 16 + i);
        end
        
        // Test alternating pattern
        $display("\n===============================================================");
        $display("Alternating Pattern Test (101010):");
        $display("--------------------------------------------------------------");
        
        for (i = 0; i < 6; i = i + 1) begin
            test_shift(i[0], 21 + i);
        end
        
        // Signal propagation verification
        $display("\n===============================================================");
        $display("Signal Propagation Analysis:");
        $display("--------------------------------------------------------------");
        
        // Reset to known state
        d = 0;
        repeat(3) @(posedge clk);
        #1;
        
        $display("Initial state: sig_a=%b, sig_b=%b, q=%b", dut.sig_a, dut.sig_b, q);
        
        d = 1;
        @(posedge clk); #1;
        $display("After 1 clock: sig_a=%b (should be 1), sig_b=%b, q=%b", dut.sig_a, dut.sig_b, q);
        
        d = 0;
        @(posedge clk); #1;
        $display("After 2 clocks: sig_a=%b, sig_b=%b (should be 1), q=%b", dut.sig_a, dut.sig_b, q);
        
        @(posedge clk); #1;
        $display("After 3 clocks: sig_a=%b, sig_b=%b, q=%b (should be 1)", dut.sig_a, dut.sig_b, q);

        
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
    
    // Task to test one shift operation
    task test_shift;
        input new_d;
        input integer cycle;
        begin
            d = new_d;
            
            // Update expected shift register
            expected_shift_reg = {expected_shift_reg[1:0], new_d};
            expected_q = expected_shift_reg[2];
            
            @(posedge clk);
            #1; // Sample after clock edge
            
            total_tests = total_tests + 1;
            if (q == expected_q) begin
                num_tests_passed = num_tests_passed + 1;
            end
            
            $display(" %2d   |  %b  | %b |   %b   |   %b   | %b |    %b     |   %03b    | %s",
                cycle, clk, d, dut.sig_a, dut.sig_b, q, expected_q, expected_shift_reg,
                (q == expected_q) ? "PASS" : "FAIL");
        end
    endtask
    
    // Monitor for observing signal changes
    initial begin
        $monitor("Time=%0t: clk=%b, d=%b, sig_a=%b, sig_b=%b, q=%b", 
            $time, clk, d, dut.sig_a, dut.sig_b, q);
    end
    
    // Generate VCD file for waveform viewing
    initial begin
        $dumpfile("top_module_shift_tb.vcd");
        $dumpvars(0, tb_top_module_shift);
        // Also dump internal flip-flop signals
        $dumpvars(1, dut.flop1);
        $dumpvars(1, dut.flop2);
        $dumpvars(1, dut.flop3);
    end
    
endmodule