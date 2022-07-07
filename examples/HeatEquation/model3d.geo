//+
Point(1) = {-1, -1, -1, 1.0};
//+
Point(2) = {1, -1, -1, 1.0};
//+
Point(3) = {1, 1, -1, 1.0};
//+
Point(4) = {-1, 1, -1, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 2} {
  Surface{1}; 
}
//+
Physical Volume("main", 33) = {1};
//+
Physical Surface("boundary", 34) = {13, 21, 26, 1, 25, 17};
