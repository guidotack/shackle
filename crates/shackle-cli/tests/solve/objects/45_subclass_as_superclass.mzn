% An object of a subclass used where its superclass is expected is the same
% object of the superclass, in every position: a declaration, a comparison, a
% function's argument and result, array, set and tuple literals, `if`, an opt
% value, a var reference, a comprehension and an index. `B` has objects from a
% root, a collection and a nested child, so they come from several places.
class A ( 1..9: x );
class B extends A ( 1..9: y );
class T ( set of new B: kids );
new A: na = (x: 1);
new B: nb = (x: 2, y: 2);
set of new B: bs = [(x: 3, y: 3)];
new T: t = (kids: [(x: 4, y: 4)]);
function int: get_x(A: a) = a.x;
function A: pick() = nb;
A: a = nb;
array [int] of A: xs = [na, nb] ++ [b | b in B];
tuple(A, int): tp = (nb, 0);
opt B: o = nb;
opt A: p = o;
var B: r;
constraint r.y = 4;
var A: s = r;
array [A] of int: w = [a: a.x * 10 | a in A];
output [
	"\(a) \(na = nb) \(nb in A) \(get_x(nb)) \(pick()) \(xs) \(tp.1) \(p)\n",
	"\(if nb.y > 1 then nb else na endif) \(nb in {na, nb}) \(s) \(w[nb]) \(sum (b in B) (w[b]))\n",
];
