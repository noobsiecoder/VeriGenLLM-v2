`timescale 1ns / 1ps

module tb_ram_8x8;
    
    // Inputs
    reg clk;
    reg we;
    reg [2:0] addr;
    reg [7:0] din;
    
    // Output
    wire [7:0] dout;
    
    // Test tracking
    reg test_passed = 1'b1;
    integer test_count = 0;
    integer pass_count = 0;
    integer i;
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Instantiate the module under test
    ram uut(
        .clk(clk),
        .we(we),
        .addr(addr),
        .din(din),
        .dout(dout)
    );
    
    // Task to write to RAM
    task write_ram;
        input [2:0] write_addr;
        input [7:0] write_data;
        begin
            @(negedge clk);
            addr = write_addr;
            din = write_data;
            we = 1'b1;
            @(posedge clk);
            @(negedge clk);
            we = 1'b0;
            #1;
            
            $display("  Write: addr=%d, data=%h", write_addr, write_data);
        end
    endtask
    
    // Task to read from RAM and check
    task read_and_check;
        input [2:0] read_addr;
        input [7:0] expected_data;
        input [127:0] test_description;
        begin
            @(negedge clk);
            addr = read_addr;
            we = 1'b0;
            @(posedge clk);
            #1; // Small delay for output to settle
            
            test_count = test_count + 1;
            $display("Test %0d: %0s", test_count, test_description);
            $display("  Read: addr=%d, expected=%h, got=%h", read_addr, expected_data, dout);
            
            if (dout === expected_data) begin
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
        $display("Problem 11: Random Access Memory (8x8)");
        $display("Description: 8 locations x 8 bits RAM");
        $display("Features:");
        $display("  - Synchronous write (on posedge clk when we=1)");
        $display("  - Synchronous read (always outputs mem[addr])");
        $display("====================================================\n");
        
        // Initialize
        clk = 0;
        we = 0;
        addr = 3'b000;
        din = 8'h00;
        
        // Wait for initial stabilization
        #20;
        
        // Test 1: Write to all locations
        $display("Test Set 1: Writing to all 8 locations");
        for (i = 0; i < 8; i = i + 1) begin
            write_ram(i[2:0], 8'h10 + i);  // Write 0x10, 0x11, ..., 0x17
        end
        
        // Test 2: Read back all locations
        $display("\nTest Set 2: Reading back all locations");
        for (i = 0; i < 8; i = i + 1) begin
            read_and_check(i[2:0], 8'h10 + i, "Read after write");
        end
        
        // Test 3: Overwrite some locations
        $display("Test Set 3: Overwriting locations");
        write_ram(3'd0, 8'hAA);
        write_ram(3'd3, 8'hBB);
        write_ram(3'd7, 8'hCC);
        
        // Read back to verify overwrites
        read_and_check(3'd0, 8'hAA, "Overwrite location 0");
        read_and_check(3'd3, 8'hBB, "Overwrite location 3");
        read_and_check(3'd7, 8'hCC, "Overwrite location 7");
        
        // Test 4: Verify other locations unchanged
        $display("Test Set 4: Verify non-overwritten locations");
        read_and_check(3'd1, 8'h11, "Location 1 unchanged");
        read_and_check(3'd2, 8'h12, "Location 2 unchanged");
        read_and_check(3'd4, 8'h14, "Location 4 unchanged");
        
        // Test 5: Test write enable functionality
        $display("Test Set 5: Testing write enable");
        @(negedge clk);
        addr = 3'd5;
        din = 8'hFF;
        we = 1'b0;  // Write disabled
        @(posedge clk);
        @(negedge clk);
        
        read_and_check(3'd5, 8'h15, "Write disabled - data unchanged");
        
        // Test 6: Back-to-back writes
        $display("Test Set 6: Back-to-back writes");
        @(negedge clk);
        addr = 3'd2;
        din = 8'h22;
        we = 1'b1;
        @(posedge clk);
        
        @(negedge clk);
        addr = 3'd3;
        din = 8'h33;
        we = 1'b1;
        @(posedge clk);
        
        @(negedge clk);
        we = 1'b0;
        
        read_and_check(3'd2, 8'h22, "Back-to-back write 1");
        read_and_check(3'd3, 8'h33, "Back-to-back write 2");
        
        // Test 7: Read during write (write-through behavior)
        $display("Test Set 7: Read during write");
        @(negedge clk);
        addr = 3'd6;
        din = 8'h66;
        we = 1'b1;
        @(posedge clk);
        #1;
        
        test_count = test_count + 1;
        if (dout === 8'h66 || dout === 8'h16) begin
            $display("Test %0d: PASS - Read during write", test_count);
            $display("  Output is %h (either new or old value acceptable)", dout);
            pass_count = pass_count + 1;
        end else begin
            $display("Test %0d: FAIL - Unexpected output during write: %h", test_count, dout);
            test_passed = 1'b0;
        end
        
        @(negedge clk);
        we = 1'b0;
        
        // Test 8: Rapid address changes (read different locations)
        $display("\nTest Set 8: Rapid address changes");
        we = 1'b0;
        
        @(negedge clk);
        addr = 3'd0;
        @(posedge clk);
        #1;
        if (dout === 8'hAA) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Quick read addr 0: %h", test_count, dout);
        end
        
        @(negedge clk);
        addr = 3'd7;
        @(posedge clk);
        #1;
        if (dout === 8'hCC) begin
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("Test %0d: PASS - Quick read addr 7: %h", test_count, dout);
        end
        
        // Test 9: Write all 1s and all 0s
        $display("\nTest Set 9: Boundary values");
        write_ram(3'd4, 8'hFF);
        write_ram(3'd5, 8'h00);
        
        read_and_check(3'd4, 8'hFF, "All 1s written correctly");
        read_and_check(3'd5, 8'h00, "All 0s written correctly");
        
        // Display test summary
        #10;
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
        #10000;
        $display("\nERROR: Timeout - Test did not complete");
        $finish;
    end
    
endmodule