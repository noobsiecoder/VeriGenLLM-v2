// Fix the latch bugs by completing the if-else structures
module top_module_always (
    input      cpu_overheated,
    output reg shut_off_computer,
    input      arrived,
    input      gas_tank_empty,
    output reg keep_driving
);

    always @(*) begin
        if (cpu_overheated)
           shut_off_computer = 1;
    end

    always @(*) begin
        if (~arrived)
           keep_driving = ~gas_tank_empty;
    end

endmodule
// Bug analysis:
// 1. First always block: Missing else clause
//    - When cpu_overheated = 0, shut_off_computer is not assigned
//    - This creates a latch (holds previous value)
//    - Fix: Add else shut_off_computer = 0;
//
// 2. Second always block: Missing else clause  
//    - When arrived = 1, keep_driving is not assigned
//    - This creates a latch
//    - Fix: Add else keep_driving = 0;
//
// Correct behavior:
// - Shut off computer only when overheated
// - Keep driving only when not arrived AND gas tank not empty