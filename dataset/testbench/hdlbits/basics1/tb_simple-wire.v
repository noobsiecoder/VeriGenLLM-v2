`timescale 1ns/1ps

module tb_top_module();
    // Signal to connect to DUT output
    reg in;
    wire out;
    
    // Instantiate the Design Under Test (DUT)
    simple_wire dut (
        .in(in), .out(out)
    );
    
    // Test variables
    reg test_passed;
    integer num_tests_passed;
    integer i;
    
    initial begin
        // Initialize test tracking
        test_passed = 1'b1;
        num_tests_passed = 0;
        
        // Display header
        $display("====================================");
        $display("Testing simple_wire");
        $display("====================================");
        $display("Time\tOutput\tExpected\tResult");
        $display("------------------------------------");
        
        // Wait for any initialization
        #10;
        
        // Test Case 1: Output follows input at 0
        in = 1'b0;
        #10;  // Wait for propagation
        if (out == in) begin
            $display("%0t\t%b\t%b\t%b\t\tPASS", $time, in, out, in);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%0t\t%b\t%b\t%b\t\tFAIL", $time, in, out, in);
            test_passed = 1'b0;
        end

        // Test Case 2: Output follows input at 1
        in = 1'b1;
        #10;
        if (out == in) begin
            $display("%0t\t%b\t%b\t%b\t\tPASS", $time, in, out, in);
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("%0t\t%b\t%b\t%b\t\tFAIL", $time, in, out, in);
            test_passed = 1'b0;
        end
        
        // Test Case 3: Multiple transitions
        $display("\nTesting multiple transitions:");
        for (i = 0; i < 10; i = i + 1) begin
            in = ~in;  // Toggle input
            #5;        // Small delay
            if (out == in) begin
                $display("%0t\t%b\t%b\t%b\t\tPASS", $time, in, out, in);
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("%0t\t%b\t%b\t%b\t\tFAIL", $time, in, out, in);
                test_passed = 1'b0;
            end
        end
        
        // Test Case 3: Multiple transitions
        $display("\nTesting multiple transitions:");
        for (i = 0; i < 10; i = i + 1) begin
            in = ~in;  // Toggle input
            #5;        // Small delay
            if (out == in) begin
                $display("%0t\t%b\t%b\t%b\t\tPASS", $time, in, out, in);
                num_tests_passed = num_tests_passed + 1;
            end else begin
                $display("%0t\t%b\t%b\t%b\t\tFAIL", $time, in, out, in);
                test_passed = 1'b0;
            end
        end

        // Test Case 4: Continuous monitoring
        $display("\nContinuous monitoring test:");
        fork
            begin
                // Drive random values
                repeat(20) begin
                    #10 in = $random;
                end
            end
            begin
                // Monitor continuously
                repeat(20) begin
                    #10;
                    if (out !== in) begin
                        $display("ERROR at time %0t: in=%b, out=%b", $time, in, out);
                        test_passed = 1'b0;
                    end
                end
            end
        join
        
        // Final Summary
        #10;
        $display("====================================");
        $display("Test Summary: %0d/%0d tests passed", num_tests_passed, num_tests_passed + (test_passed ? 0 : 1));
        if (test_passed)
            $display("Overall Result: ALL TESTS PASSED");
        else
            $display("Overall Result: SOME TESTS FAILED");
        $display("====================================");
        
        // End simulation
        #10;
        $finish;
    end
    
    // Optional: Generate VCD file for waveform viewing
    initial begin
        $dumpfile("simple_wire_tb.vcd");
        $dumpvars(0, tb_top_module);
    end
    
endmodule