`timescale 1ns / 1ps

module tb_priority_encoder_3bit;
    
    // Inputs
    reg [2:0] in;
    
    // Outputs
    wire [1:0] out;
    wire valid;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    
    // Instantiate the module under test
    priority_encoder_3bit uut(
        .in(in),
        .out(out),
        .valid(valid)
    );
    
    // Task to check encoder output
    task check_encoder;
        input [2:0] test_in;
        input [1:0] expected_out;
        input expected_valid;
        input [127:0] test_description;
        begin
            in = test_in;
            #10; // Wait for propagation
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Input: in=%b", in);
            $display("  Expected: out=%b, valid=%b", expected_out, expected_valid);
            $display("  Got:      out=%b, valid=%b", out, valid);
            
            if (out === expected_out && valid === expected_valid) begin
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
        $display("Problem 4: 3-bit Priority Encoder");
        $display("Description: Priority encoder with highest priority at in[2]");
        $display("Priority: in[2] > in[1] > in[0]");
        $display("====================================================\n");
        
        // Test all possible input combinations
        $display("Running exhaustive tests (all 8 combinations):\n");
        
        // Test 1: No input active
        check_encoder(3'b000, 2'b00, 1'b0, "No input active");
        
        // Test 2: Only in[0] active (lowest priority)
        check_encoder(3'b001, 2'b00, 1'b1, "Only in[0] active");
        
        // Test 3: Only in[1] active (middle priority)
        check_encoder(3'b010, 2'b01, 1'b1, "Only in[1] active");
        
        // Test 4: in[0] and in[1] active (in[1] wins)
        check_encoder(3'b011, 2'b01, 1'b1, "in[0] and in[1] active");
        
        // Test 5: Only in[2] active (highest priority)
        check_encoder(3'b100, 2'b10, 1'b1, "Only in[2] active");
        
        // Test 6: in[0] and in[2] active (in[2] wins)
        check_encoder(3'b101, 2'b10, 1'b1, "in[0] and in[2] active");
        
        // Test 7: in[1] and in[2] active (in[2] wins)
        check_encoder(3'b110, 2'b10, 1'b1, "in[1] and in[2] active");
        
        // Test 8: All inputs active (in[2] wins)
        check_encoder(3'b111, 2'b10, 1'b1, "All inputs active");
        
        // Additional dynamic test
        $display("Dynamic test - changing inputs:");
        in = 3'b001;
        #10;
        $display("Initial: in=%b, out=%b, valid=%b", in, out, valid);
        
        in = 3'b011;  // Add in[1]
        #10;
        $display("Add in[1]: in=%b, out=%b, valid=%b", in, out, valid);
        
        in = 3'b111;  // Add in[2]
        #10;
        $display("Add in[2]: in=%b, out=%b, valid=%b", in, out, valid);
        
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
    
    // Timeout
    initial begin
        #1000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule