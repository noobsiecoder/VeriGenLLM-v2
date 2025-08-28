`timescale 1ns/1ps

module tb_top_module_always();
    // Inputs
    reg cpu_overheated;
    reg arrived;
    reg gas_tank_empty;
    
    // Outputs
    wire shut_off_computer;
    wire keep_driving;
    
    // Instantiate DUT
    top_module_always dut (
        .cpu_overheated(cpu_overheated),
        .shut_off_computer(shut_off_computer),
        .arrived(arrived),
        .gas_tank_empty(gas_tank_empty),
        .keep_driving(keep_driving)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing Latch-Free Control Logic");
        $display("1. Shut off computer when overheated");
        $display("2. Keep driving if not arrived AND not empty tank");
        $display("===============================================================");
        
        // Test 1: Complete truth table
        $display("\nTest 1: Complete Truth Table");
        $display("---------------------------------------------------------------");
        $display("cpu_oh | arrived | gas_empty | shut_off | keep_driving | Result");
        $display("---------------------------------------------------------------");
        
        for (i = 0; i < 8; i = i + 1) begin
            {cpu_overheated, arrived, gas_tank_empty} = i[2:0];
            #10;
            
            total_tests = total_tests + 1;
            
            // Check shut_off_computer logic
            if (shut_off_computer == cpu_overheated) begin
                // Check keep_driving logic
                if (arrived) begin
                    // If arrived, should stop driving
                    if (keep_driving == 0) begin
                        num_tests_passed = num_tests_passed + 1;
                        $display("   %b   |    %b    |     %b     |    %b     |      %b       | PASS",
                            cpu_overheated, arrived, gas_tank_empty, 
                            shut_off_computer, keep_driving);
                    end else begin
                        $display("   %b   |    %b    |     %b     |    %b     |      %b       | FAIL",
                            cpu_overheated, arrived, gas_tank_empty, 
                            shut_off_computer, keep_driving);
                    end
                end else begin
                    // If not arrived, keep driving unless tank empty
                    if (keep_driving == ~gas_tank_empty) begin
                        num_tests_passed = num_tests_passed + 1;
                        $display("   %b   |    %b    |     %b     |    %b     |      %b       | PASS",
                            cpu_overheated, arrived, gas_tank_empty, 
                            shut_off_computer, keep_driving);
                    end else begin
                        $display("   %b   |    %b    |     %b     |    %b     |      %b       | FAIL",
                            cpu_overheated, arrived, gas_tank_empty, 
                            shut_off_computer, keep_driving);
                    end
                end
            end else begin
                $display("   %b   |    %b    |     %b     |    %b     |      %b       | FAIL",
                    cpu_overheated, arrived, gas_tank_empty, 
                    shut_off_computer, keep_driving);
            end
        end
        
        // Test 2: Specific scenarios
        $display("\nTest 2: Specific Scenarios");
        $display("---------------------------------------------------------------");
        
        // Normal operation
        cpu_overheated = 0; arrived = 0; gas_tank_empty = 0;
        #10;
        $display("Normal driving: CPU OK, not arrived, has gas");
        $display("  shut_off=%b (should be 0), keep_driving=%b (should be 1)",
            shut_off_computer, keep_driving);
        
        // Overheated while driving
        cpu_overheated = 1; arrived = 0; gas_tank_empty = 0;
        #10;
        $display("\nOverheated while driving:");
        $display("  shut_off=%b (should be 1), keep_driving=%b (should be 1)",
            shut_off_computer, keep_driving);
        
        // Need to refuel
        cpu_overheated = 0; arrived = 0; gas_tank_empty = 1;
        #10;
        $display("\nNeed to refuel:");
        $display("  shut_off=%b (should be 0), keep_driving=%b (should be 0)",
            shut_off_computer, keep_driving);
        
        // Arrived at destination
        cpu_overheated = 0; arrived = 1; gas_tank_empty = 0;
        #10;
        $display("\nArrived at destination:");
        $display("  shut_off=%b (should be 0), keep_driving=%b (should be 0)",
            shut_off_computer, keep_driving);
        
        // Test 3: Latch detection
        $display("\nTest 3: Checking for Latches (Toggle Test)");
        $display("---------------------------------------------------------------");
        
        // Set all inputs to 1
        cpu_overheated = 1; arrived = 1; gas_tank_empty = 1;
        #10;
        $display("All inputs = 1: shut_off=%b, keep_driving=%b", 
            shut_off_computer, keep_driving);
        
        // Toggle to all 0
        cpu_overheated = 0; arrived = 0; gas_tank_empty = 0;
        #10;
        $display("All inputs = 0: shut_off=%b, keep_driving=%b", 
            shut_off_computer, keep_driving);
        
        // If outputs changed appropriately, no latches exist
        if (shut_off_computer == 0 && keep_driving == 1) begin
            $display("Outputs changed correctly - No latches detected: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Outputs stuck - Possible latch: FAIL");
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
    
    // Generate VCD file
    initial begin
        $dumpfile("latch_fix_tb.vcd");
        $dumpvars(0, tb_top_module_always);
    end
    
endmodule