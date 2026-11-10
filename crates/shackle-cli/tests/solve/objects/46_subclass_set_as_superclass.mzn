% A set of objects of a subclass used where a set of its superclass is
% expected is the same objects of the superclass, par and var: an argument of
% `in` and `subset`, and a declaration.
class A ( 1..9: x );
class B extends A ( 1..9: y );
new A: na = (x: 1);
set of new B: bs = [(x: 2, y: 2), (x: 3, y: 3)];
var set(0..2) of new B: vbs;
constraint card(vbs) = 1;
constraint forall (b in vbs) (b.y = 5);
var set of A: s = B;
var set of A: vs = vbs;
output ["\(na in B) \(B subset A) \(s) \(card(vs)) \(vs subset A)\n"];
