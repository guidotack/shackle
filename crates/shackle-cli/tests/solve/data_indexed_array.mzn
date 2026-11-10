% Indexed array literals in data: the elements are placed by their indices, in
% any order, and each index set runs from the smallest to the largest index.
enum Colour = {Red, Green, Blue};
array [Colour] of int: w;
array [int, int] of int: m;
array [1..2, Colour] of int: h;
var 0..20: x;
constraint x = w[Green] + m[3, 2] + h[1, Blue];
output ["\(w) \(index_set_1of2(m)) \(index_set_2of2(m)) \(m) \(h) \(x)\n"];
