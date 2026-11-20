% Sets of bools in an output item, on their own and in an array. A set of bools
% can't be iterated over, so it has its own `show` in shackle's library. Only
% the MicroZinc target uses that library.
set of bool: b1 = {false, true};
set of bool: b2 = {true};
set of bool: b3 = {false};
output ["\(b1) \(b2) \(b3) \([{true}, {}])\n"];
