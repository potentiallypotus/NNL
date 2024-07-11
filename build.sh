#!/bin/sh

clang++ -pedantic -std="c++17" NNL.cpp main.cpp -O0 -g -o main
./main
