% Every solution once: an absent optional variable has one representation, so
% the value it would have doesn't give extra solutions, also without a domain
% (bool) and in an array or a tuple. `b` and `ab[1]` give 3 * 2 solutions. (An
% unbounded absent `var opt int` is covered by erase_opt's unit tests: with the
% bug, it would have infinitely many solutions.)
var opt bool: b;
array [1..2] of var opt bool: ab;
constraint occurs(ab[1]) -> deopt(ab[1]);
constraint absent(ab[2]);
var tuple(opt 1..2, opt bool): t;
constraint absent(t.1) /\ absent(t.2);
output ["\(b) \(ab) \(t)\n"];
