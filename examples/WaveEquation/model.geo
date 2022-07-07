// Gmsh project created on Mon Jul  4 16:01:00 2022
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 0.1, 0, 2*Pi};
//+
Ellipse(2) = {0, 0, 0, 1, 0.9, 0, 2*Pi};
//+
Curve Loop(1) = {2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1};
//+
Surface(2) = {2};
//+
Recursive Delete {
  Surface{2}; 
}
//+
Recursive Delete {
  Surface{1}; 
}
//+
Circle(1) = {0, -0, 0, 0.1, 0, 2*Pi};
//+
Ellipse(2) = {0, 0, 0, 1, 0.9, 0, 2*Pi};
//+
Curve Loop(3) = {2};
//+
Curve Loop(4) = {1};
//+
Surface(1) = {3, 4};
//+
Curve Loop(5) = {2};
//+
Curve Loop(6) = {1};
//+
Plane Surface(1) = {5, 6};
//+
Curve Loop(7) = {1};
//+
Surface(2) = {7};
//+
Physical Curve("boundary", 9) = {2};
//+
Physical Surface("main", 10) = {1, 2};
//+
Physical Surface("source", 11) = {2};
