% Three levels of inheritance: the objects of `C` are objects of `B` and of `A`,
% with a singular root, a collection root and nested objects of `C`, and a
% union of a set of `B` and a set of `C` used as a set of `A`.
class A ( 1..9: x );
class B extends A ( 1..9: y );
class C extends B ( 1..9: z );
class T ( set of new C: kids );
new C: nc = (x: 1, y: 1, z: 1);
set of new B: nbs = [(x: 2, y: 2), (x: 3, y: 3)];
set of new C: ncs = [(x: 4, y: 4, z: 4)];
new T: t = (kids: [(x: 5, y: 5, z: 5)]);
set of B: bs = {b | b in B where b.x = 2};
set of C: cs = {c | c in C where c.z >= 4};
set of A: as = bs union cs;
A: a = nc;
output [
	"\(A) \(B) \(C)\n",
	"\(as) \(a) \(nc in B) \(sum (a in A) (a.x)) \(sum (b in B) (b.y)) \(sum (c in C) (c.z))\n",
];
