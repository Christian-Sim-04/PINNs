//+
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 185, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 100, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {2};
//+
Plane Surface(1) = {1, 2};
