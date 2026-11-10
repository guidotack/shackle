% Arrays of two and more dimensions, read back from the solver without an output
% item (so as values), and given in JSON data.
array [1..2, 1..3] of int: d;
array [1..2, 1..2] of var opt 1..2: o;
array [1..2, 1..2] of var opt bool: b;
array [1..2, 1..2, 1..2] of var 0..9: w;
constraint forall (i, j in 1..2) (occurs(o[i, j]) <-> i = j);
constraint forall (i in 1..2) (deopt(o[i, i]) = i);
constraint forall (i, j in 1..2) (occurs(b[i, j]) <-> i != j);
constraint forall (i in 1..2) (deopt(b[i, 3 - i]) = (i = 1));
constraint forall (i, j, k in 1..2) (w[i, j, k] = d[i, j + k - 1]);
