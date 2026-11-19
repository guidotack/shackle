% A 1-tuple is shown with a trailing comma, also when its field is optional (and
% absent): `show` is generated for it when the opt type is erased.
tuple(opt int): t = (<>,);
tuple(opt int): u = (3,);
tuple(int): v = (3,);
tuple(opt int, int): w = (<>, 1);
tuple(tuple(opt bool)): n = ((<>,),);
var tuple(opt 1..2): x;
constraint absent(x.1);
output ["\(t) \(u) \(v) \(w) \(n) \(x)\n"];
