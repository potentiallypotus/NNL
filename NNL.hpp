#ifndef NNL_HPP
#define NNL_HPP
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>
#include <algorithm>
#include <memory>

namespace{
    void srandf();
    void srandf(int);
    float randf();
};

typedef std::vector<float> list;
typedef std::vector<list> matrix;
typedef struct dataPoint{
    list inputs;
    list outputs;
} DataPoint;
typedef std::vector<DataPoint> dataSet;

class Neuron;


struct model{
    unsigned int numIns;
    unsigned int numOuts;
    unsigned int numNeurons;
    unsigned int numLayers;
    std::vector<DataPoint> trainingSet;
    std::vector<unsigned int> shape;
    list weightsList;
    list biasList;
    std::vector<std::vector<Neuron*>> NMat;
    std::vector<matrix> wMats;
    matrix aMat;
    matrix bMat;
    
    model(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape);
    model();
};

class NN{
    //Members
    public:
    struct model m{};
    std::vector<Neuron> neuronList;
    std::vector<Neuron> inputLayer;
    std::vector<Neuron*> layerList;
    std::vector<float> expectedOuts;
    float gradientDescent();
    public:
    float dCost(float* toMod);
    list forward(list ins, list eOuts);
    NN(struct model& m, float(*activationFunction)(float));
    NN(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape, float(*activationFunction)(float));
    float cost();
    float train(unsigned int iterations);
    void initializeParams();
    void test();
    void backProp(std::vector<matrix>& weightsMatricies, matrix& biasMatrix, matrix& activatedMatrix, float rate);
    void printState();
    friend class Neuron;


};
void test(NN);
#endif //NNL_HPP





