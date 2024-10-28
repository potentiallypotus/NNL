#include <iostream>
#include "linear.cuh"
#include "NNL.hpp"
#include <vector>
#include <cmath>

int main(void){
    std::vector<float> trainingInput;
    std::vector<float> trainingOutput;
    for (float i = -10.0f; i < 10.0f; i+=1.0f){
        trainingInput.push_back(i);
        trainingOutput.push_back(sin(i)+(i/2));
    }
    const unsigned int numLayers = 4;
    unsigned int* shape = new unsigned int[numLayers];
    shape[0] = 1;
    shape[1] = 10;
    shape[2] = 10;
    shape[3] = 1;
    Network net(numLayers, shape);
    float*& input = net.pointerToLayers[0].valueArray.hostData;
    for (int i = 0; i < trainingInput.size(); i++){
        input[0] = trainingInput[i];
        net.forward();
        std::cout<<"dataPoint "<<i<<": input: "<<trainingInput[i]<<" ~~~~~ expected output: "<<trainingOutput[i]<<std::endl;
        std::cout<<"predicted output:\n";
        net.pointerToLayers[numLayers-1].valueArray.print();
        std::cout<<std::endl<<std::endl;
    }
}
