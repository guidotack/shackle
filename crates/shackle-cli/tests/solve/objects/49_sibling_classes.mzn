% Objects of sibling classes (and of a subclass of a sibling) used together:
% their type is their nearest common superclass `A`, in an array or set literal,
% `if`, `in`, `union`, a comparison, and a comprehension over a var reference.
class A (1..9: x);
class B extends A ();
class C extends A ();
class D extends B ();
new B: nb = (x: 1);
new C: nc = (x: 2);
new D: nd = (x: 3);
set of new C: cs = [(x: 4), (x: 5)];
var C: r;
constraint r.x = 5;
output [
	"\([nb, nc, nd]) \(if nb.x > 5 then nb else nc endif) \(nd in C) \(nc in B union C)\n",
	"\(nd = nc) \(nb != nd) \({nb, nc}) \(r = nd) \(r = nc) \(sum (a in [nd, r]) (a.x))\n",
];
