`timescale 1ns/1ps

module tb_top_module_always_case();
    // Inputs
    reg [2:0] sel;
    reg [3:0] data0, data1, data2, data3, data4, data5;
    
    // Output
    wire [3:0] out;
    
    // Instantiate DUT
    top_module_always_case dut (
        .sel(sel),
        .data0(data0), .data1(data1), .data2(data2),
        .data3(data3), .data4(data4), .data5(data5),
        .out(out)
    );
    
    // Test variables
    integer num_tests_passed;
    integer total_tests;
    integer i;
    reg [3:0] expected_out;
    
    initial begin
        // Initialize
        num_tests_passed = 0;
        total_tests = 0;
        
        // Display header
        $display("===============================================================");
        $display("Testing 6-to-1 Multiplexer with Case Statement");
        $display("sel 0-5: Select corresponding data input");
        $display("sel 6-7: Output 0");
        $display("===============================================================");
        
        // Initialize data inputs with distinct values
        data0 = 4'hA;
        data1 = 4'hB;
        data2 = 4'hC;
        data3 = 4'hD;
        data4 = 4'hE;
        data5 = 4'hF;
        
        // Test 1: Test all valid selections
        $display("\nTest 1: Valid Selection Tests (sel = 0 to 5)");
        $display("-----------------------------------------------");
        $display("sel | Expected | Actual | Result");
        $display("-----------------------------------------------");
        
        for (i = 0; i <= 5; i = i + 1) begin
            sel = i[2:0];
            
            case (sel)
                3'd0: expected_out = data0;
                3'd1: expected_out = data1;
                3'd2: expected_out = data2;
                3'd3: expected_out = data3;
                3'd4: expected_out = data4;
                3'd5: expected_out = data5;
                default: expected_out = 4'b0;
            endcase
            
            #10;
            
            total_tests = total_tests + 1;
            if (out == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
                $display(" %0d  |    %h     |   %h    | PASS", sel, expected_out, out);
            end else begin
                $display(" %0d  |    %h     |   %h    | FAIL", sel, expected_out, out);
            end
        end
        
        // Test 2: Test invalid selections (should output 0)
        $display("\nTest 2: Invalid Selection Tests (sel = 6, 7)");
        $display("-----------------------------------------------");
        
        for (i = 6; i <= 7; i = i + 1) begin
            sel = i[2:0];
            expected_out = 4'b0;
            #10;
            
            total_tests = total_tests + 1;
            if (out == expected_out) begin
                num_tests_passed = num_tests_passed + 1;
                $display(" %0d  |    %h     |   %h    | PASS", sel, expected_out, out);
            end else begin
                $display(" %0d  |    %h     |   %h    | FAIL", sel, expected_out, out);
            end
        end
        
        // Test 3: Dynamic data change test
        $display("\nTest 3: Dynamic Data Change Test");
        $display("-----------------------------------------------");
        
        sel = 3'd2;  // Select data2
        $display("Selecting data2 (sel = 2):");
        
        data2 = 4'h5;
        #10;
        $display("  data2 = %h, out = %h (should be %h)", data2, out, data2);
        check_result(data2, "Dynamic update 1");
        
        data2 = 4'h9;
        #10;
        $display("  data2 = %h, out = %h (should be %h)", data2, out, data2);
        check_result(data2, "Dynamic update 2");
        
        // Test 4: All inputs same value test
        $display("\nTest 4: All Inputs Same Value Test");
        $display("-----------------------------------------------");
        
        data0 = 4'h7; data1 = 4'h7; data2 = 4'h7;
        data3 = 4'h7; data4 = 4'h7; data5 = 4'h7;
        
        for (i = 0; i <= 5; i = i + 1) begin
            sel = i[2:0];
            #10;
            if (out == 4'h7) begin
                num_tests_passed = num_tests_passed + 1;
            end
            total_tests = total_tests + 1;
        end
        $display("All selections with same data: %s", 
            (out == 4'h7) ? "PASS" : "FAIL");
        
        // Test 5: Rapid selection changes
        $display("\nTest 5: Rapid Selection Changes");
        $display("-----------------------------------------------");
        
        // Reset data to distinct values
        data0 = 4'h0; data1 = 4'h1; data2 = 4'h2;
        data3 = 4'h3; data4 = 4'h4; data5 = 4'h5;
        
        $display("Rapid switching through all inputs:");
        for (i = 0; i <= 7; i = i + 1) begin
            sel = i[2:0];
            #2;  // Short delay
            
            if (i <= 5) begin
                expected_out = i[3:0];
            end else begin
                expected_out = 4'b0;
            end
            
            $display("  sel=%0d: out=%h (expected %h)", sel, out, expected_out);
        end
        
        // Test 6: Width verification
        $display("\nTest 6: Output Width Verification");
        $display("-----------------------------------------------");
        
        // Test with all bits set
        data3 = 4'b1111;
        sel = 3'd3;
        #10;
        
        if (out == 4'b1111) begin
            $display("Full width output test: PASS");
            num_tests_passed = num_tests_passed + 1;
        end else begin
            $display("Full width output test: FAIL (out = %b)", out);
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
    
    // Task to check result
    task check_result;
        input [3:0] expected;
        input [20*8:1] description;
        begin
            total_tests = total_tests + 1;
            
            if (out == expected) begin
                num_tests_passed = num_tests_passed + 1;
                $display("  Result: PASS (%0s)", description);
            end else begin
                $display("  Result: FAIL (%0s)", description);
            end
        end
    endtask
    
    // Generate VCD file
    initial begin
        $dumpfile("mux6to1_tb.vcd");
        $dumpvars(0, tb_top_module_always_case);
    end
    
endmodule