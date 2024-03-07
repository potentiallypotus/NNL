#ifndef NNL_HPP
#define NNL_HPP
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>
#include <algorithm>

class Layer;
class NN;


typedef size_t Uint;
typedef std::vector<double> list;
typedef std::vector<list> matrix;
typedef std::vector<matrix> matrix3;
typedef struct dataPoint{
    list inputs;
    list outputs;
} DataPoint;
typedef std::vector<DataPoint> dataSet;
using actF = double(*)(double);
enum activationFunctionIndex {
    sig = 0,
    relu = 2,
    bin = 4,
    lin = 6,
    tanH = 8
};

namespace{
    void srandf();
    void srandf(int);
    float randf();
};


struct model{
    std::vector<DataPoint> trainingSet;
    std::vector<Uint> shape;//shape of hidden layers and output layers
    Uint numLayers;
    Uint numIns;
    Uint numOuts;
    //[layer L ][index j ]
    matrix biases;
    //[layer l][index j][previous layer index k]
    matrix3 weights;
    
    model(const std::vector<DataPoint> trainingSet, const std::vector<Uint> shape);
    model();
};

class NN{
    public:
    double rate;    
    struct model m;
    std::vector<activationFunctionIndex> AFI;

    matrix a;
    matrix z;
    list expectedOuts;
    

    Uint numIns;
    Uint numOuts;
    Uint numNeurons;
    Uint numLayers;


    matrix3 weightDelta;
    matrix bDelta;
    matrix dCda;
    void initDeltas();
    static const actF activationFunction[10];
    list forward(list ins, list eOuts);

    void train(Uint epoch);
    void gradientDescent();
    void backProp();
    void resetTemps();
    void test();
    void printState();
    NN(struct model& m, double rate);
    
    float cost();
    friend class Layer;
};
void test(NN);
#endif //NNL_HPP





