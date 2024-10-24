#include <iostream>
#include "linear.cuh"


int main(void){
    Matrix a(3, 3);
    Matrix b(3, 3);
    for (int i = 0; i < 9; i++){
        a.hostData[i] = 3.0f;
        b.hostData[i] = 3.9f;
    }
    a.allocate();
    b.allocate();
    a.copyToDevice();
    b.copyToDevice();
    a.add(b);
    for (int i = 0; i< 9; i++){
        std::cout<<a.hostData[i]<< " ";
    }
}
