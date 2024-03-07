#!/bin/sh

clang++ -Wall -pedantic -std="c++17" NNL.cpp main.cpp -o main -O0 -g && ./main
