% A superclass's set of objects is a var set when a subclass has a var root,
% and is then built per object. The objects of a root that comes after another
% root of the same class are numbered after it (`B_occ_k(<start>..<end> - 1)`),
% not from 1, and must still be in the superclass's set: `b2` and `b3` after
% `vb`, and `bs` after the objects of `C` (three levels).
class A ( 1..9: x );
class B extends A ( 1..9: y );
class C extends B ( 1..9: z );
var set(0..1) of new B: vb;
set of new B: b2 = [(x: 2, y: 2)];
set of new B: b3 = [(x: 3, y: 3), (x: 4, y: 4)];
var set(0..1) of new C: vc;
set of new B: bs = [(x: 5, y: 5)];
constraint card(vb) = 1 /\ forall (b in vb) (b.x = 1 /\ b.y = 1);
constraint card(vc) = 1 /\ forall (c in vc) (c.x = 6 /\ c.y = 6 /\ c.z = 6);
output ["\(A) \([a.x | a in A]) \(B) \(sum (b in B) (b.y)) \(C)\n"];
