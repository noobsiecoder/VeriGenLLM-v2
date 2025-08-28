// This is a 7458 microcontroller problem
module microcontroller_7458 ( 
    input p1a, p1b, p1c, p1d, p1e, p1f,
    output p1y,
    input p2a, p2b, p2c, p2d,
    output p2y
);

    wire W, X, Y, Z; // assign output signals

    assign W = p1a & p1b & p1c;
    assign X = p2a & p2b;

    assign Y = p1d & p1e & p1f;
    assign Z = p2c & p2d;

    assign p1y = W | Y; // output signal at P1
    assign p2y = X | Z; // output signal at P2

endmodule