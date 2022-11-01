function abg = rt2abg(worldOrientation, worldLocation)
R = worldOrientation';
T = worldLocation;

syms a b g

r_x = [[1      0      0 ];
       [0  cos(a) -sin(a)];
       [0  sin(a) cos(a)]];
r_y = [[cos(b) 0 sin(b)];
       [0      1      0];
       [-sin(b) 0 cos(b)]];
r_g = [[cos(g) -sin(g) 0];
       [sin(g)  cos(g) 0];
       [ 0      0      1]];

bag = r_y*r_x*r_g;

g_=atan(R(2,1)/R(2,2));
b_=atan(R(1,3)/R(3,3));
a_=asin(-R(2,3));

% a_2 = a_;
% b_2 = b_;
% g_2 = g_;

% a_ = pi-a_;
% b_ = b_+pi;
% g_ = g_+ pi;

% a_
% b_
% g_

soln = subs(bag, [a,b,g], [a_,b_,g_]);
soln = simplify(soln);
check = vpa(soln, 9)';
A = check.*worldOrientation;

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
end

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    b_ = b_+pi;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    b_ = b_-pi;
end

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    g_ = g_+ pi;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    g_ = g_- pi;
end

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    b_ = b_+pi;
    g_ = g_+ pi;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    b_ = b_-pi;
    g_ = g_- pi;
end

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    g_ = g_+ pi;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    g_ = g_- pi;
end

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    b_ = b_+pi;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    b_ = b_-pi;
end

if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    b_ = b_+pi;
    g_ = g_+ pi;
    soln = subs(bag, [a,b,g], [a_,b_,g_]);
    soln = simplify(soln);
    check = vpa(soln, 9)';
    A = check.*worldOrientation;
end
if A(1)<=0 || A(2)<=0 || A(3)<=0 || A(4)<=0 || A(5)<=0 || A(6)<=0 || A(7)<=0 || A(8)<=0 || A(9)<=0
    a_ = pi-a_;
    b_ = b_-pi;
    g_ = g_- pi;
end

syms r trans_x trans_z
R = check';
zz = [R(1,3),R(2,3),R(3,3)];
yy = [R(1,2),R(2,2),R(3,2)];
xx = [R(1,1),R(2,1),R(3,1)];

x = r*-zz(1);
y = r*-zz(2);
z = r*-zz(3);

pos = [x y z]+trans_x*xx+trans_z*yy;

eqns = pos == T;

S = solve(eqns, [r trans_x trans_z]);

r_ = double(S.r);
trans_x_ = double(S.trans_x);
trans_z_ = double(S.trans_z);

abg = [a_,b_,g_,trans_x_,trans_z_,r_];