// Gmsh project created on Fri Aug  5 18:11:30 2022
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2.2, 0, 0, 1.0};
//+
Point(3) = {2.2, 0.41, 0, 1.0};
//+
Point(4) = {0, 0.41, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Circle(5) = {0.2, 0.2, 0, 0.05, 0, 2*Pi};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Curve Loop(2) = {5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("no-slip boundary", 6) = {3, 1, 5};
//+
Physical Curve("left", 7) = {4};
//+
Physical Surface("main", 8) = {1};
