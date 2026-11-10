% An object of a subclass where its superclass is expected goes through the map
% `B_to_A`, one block per place that creates `B` objects; a set of them through
% `mzn_injective_image`.
class A ( 1..9: x );
class B extends A ( 1..9: y );
class T ( set of new B: kids );
new B: nb = (x: 2, y: 2);
set of new B: bs = [(x: 3, y: 3)];
new T: t = (kids: [(x: 4, y: 4)]);
A: a = nb;
set of A: s = B;
solve satisfy;
