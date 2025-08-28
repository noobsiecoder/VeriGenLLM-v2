// This is a 7458 microcontroller problem
module microcontroller_7458 ( 
    input p1a, p1b, p1c, p1d, p1e, p1f,
    output p1y,
    input p2a, p2b, p2c, p2d,
    output p2y );
// Consider a 7458 microcontroller. It has four AND gates and two OR gates
// Create a chip that has 10 inputs and 2 outputs following these conditions:
// 1) p1a, p1c, p1b passing signal via an AND gate  = W
// 2) p2a, p2b passing signal via an AND gate       = X
// 3) p1f, p1e, p1d passing signal via an AND gate  = Y
// 4) p2c, p2d passing signal via an AND gate       = Z
// Insert code here