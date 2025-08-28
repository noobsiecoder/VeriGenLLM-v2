// Fix the latch bugs in this control logic
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
// This code has incomplete if statements that create latches
// Ensure all outputs are assigned in all cases
// shut_off_computer should be 0 when CPU is not overheated
// keep_driving should be 0 when arrived at destination