
# About

This is a program of NMF(NTF) (Non-negative Matrix(Tensor) Factorization) based on C++ and Eigen library

## How to Use

Put [ Eigen library](http://eigen.tuxfamily.org/index.php) on the same directory as this program.

compile a file like `pntf_eigen.cpp` with compile options `-std=c++1y -I .`

## Requirement

Input file must be Tab separated file.

### Case of NMF

Input file must have description on the first line.
The first row of every line have also must have description.
The other data should is as a matrix.

### Case of NTF

Input file are like as NMF.
The different point is that columns are the product of two factors.
The program read columns iteratively as you mentioned in arguments.
