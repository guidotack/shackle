% Sets of enum values in the solution, read back from the solver: on their own,
% in an array, a tuple, a record and an opt set. The MicroZinc target erases
% enums, so the sets come back as sets of integers and must be mapped back. No
% two members are adjacent, as the targets show adjacent members differently
% (A37).
enum Colour = {Red, Green, Blue};
var set of Colour: vs;
constraint vs = {Red, Blue};
array [1..2] of var set of Colour: avs;
constraint avs[1] = {Green} /\ avs[2] = {};
var tuple(set of Colour, int): t;
constraint t.1 = {Red, Blue} /\ t.2 = 1;
var record(set of Colour: s): r;
constraint r.s = {Blue};
var opt set of Colour: os;
constraint occurs(os) /\ deopt(os) = {Red};
