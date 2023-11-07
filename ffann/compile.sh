#!bin/sh
# The following compiles newffann.c (which includes newaux.c) including linking to math.h
gcc -fopenmp algorithm.c -lm -o  algorithm.exe 
