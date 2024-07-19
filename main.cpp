#include "NNL.hpp"

#define EPOCH 10
#define RATE 1e-0

dataSet buildBinSet(){
    int numNodes = 4;
    int numVals = 16;
    dataSet bin;
    for (unsigned int i = 0; i< numVals; ++i){
        list ins;
        list out = {(float)i};
        unsigned int val = i;
        for (int j = 1; j<=numNodes; j++){
            if(val>=(1<<(numNodes-j))){
                ins.push_back(1);
                val -= 1<<(numNodes-j);
            }else{
                ins.push_back(0);
            }
            
        }
        DataPoint point = {ins,out};
        bin.push_back(point);
    }
    return bin;
}

dataSet buildAddSet(){
    // int numNodes = 4;
    // int numVals = 8;
    dataSet add = dataSet(10);
    for (dataPoint dp : add){
    }
    return add;
}

int main(void){
    int iterations = EPOCH;
    if (iterations < 0){iterations *= -1;}
    std::vector<DataPoint> XOR = {
        {{0,0}, {0}},
        {{1,0}, {1}},
        {{0,1}, {1}},
        {{1,1}, {0}}
    };
    // std::vector<DataPoint> OR = {
    //     {{0,0}, {0}},
    //     {{1,0}, {1}},
    //     {{0,1}, {1}},
    //     {{1,1}, {1}}
    // };
    // std::vector<DataPoint> SQRT = {
    //     {{0},{0}},
    //     {{2},{(float)sqrt(2)}},
    //     {{4},{2}},
    //     {{9},{3}},
    // };
    //std::vector<DataPoint> BIN = buildBinSet();

    std::vector<Uint> shape = {2,1};
    
    struct model m = model(XOR, shape);
    NN network = NN(m, RATE);
    //std::cout<<"debug1\n";  
    std::cout<<"cost: " << network.cost()<<std::endl;
    network.test();
    //network.printState();
    network.train(iterations);
    std::cout<<"cost: " << network.cost() <<std::endl;
    network.test();
    network.printState();

}