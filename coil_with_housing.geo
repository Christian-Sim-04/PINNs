Merge "coil_with_housing.step";
//+
Curve Loop(70) = {7};
//+
Curve Loop(71) = {5};
//+
Curve Loop(72) = {3};
//+
Plane Surface(64) = {70, 71, 72};
//+
Surface Loop(16) = {8, 5, 6, 7};
//+
Surface Loop(17) = {40, 41, 42};
//+
Surface Loop(18) = {37, 38, 39};
//+
Volume(16) = {16, 17, 18};
//+
Surface Loop(19) = {19, 20, 27, 10, 22, 9, 26, 25, 24, 21, 18, 17, 16, 15, 14, 13, 12, 11, 23};
//+
Surface Loop(20) = {43, 44, 45};
//+
Surface Loop(21) = {28, 29, 30};
//+
Surface Loop(22) = {61, 62, 63};
//+
Surface Loop(23) = {31, 32, 33};
//+
Surface Loop(24) = {34, 35, 36};
//+
Volume(17) = {16, 17, 18, 19, 20, 21, 22, 23, 24};
//+
Volume(18) = {16, 19};
