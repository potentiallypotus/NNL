#ifndef NNL_HPP
#define NNL_HPP
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>
#include <algorithm>

namespace{
    void srandf();
    void srandf(int);
    float randf();
};

typedef struct dataPoint{
    std::vector<float> inputs;
    std::vector<float> outputs;
} DataPoint;
struct model{
    unsigned int numIns;
    unsigned int numOuts;
    unsigned int numNeurons;
    std::vector<DataPoint> trainingSet;
    std::vector<unsigned int> shape;
    std::vector<float> weightsList;
    std::vector<float> biasList;
    
    model(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape);
    model();
};
class Neuron{
    public:
    static unsigned int nextId;
    unsigned int id;
    std::vector<float*> wptr;
    float* b;
    float val;
    float(*activate)(float);

    public:
    Neuron();
    Neuron(unsigned int numInputs, float(*activationFunction)(float), unsigned int& wCursor, model& model);
    float update(std::vector<float> neuronInputs);
    friend class NN;


};
class NN{
    //Members
    public:
    struct model m{};
    std::vector<Neuron> neuronList;
    float gradientDescent();
    public:
    float dCost(float* toMod);
    std::vector<float> forward(std::vector<float> ins);
    NN(struct model& m, float(*activationFunction)(float));
    NN(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape, float(*activationFunction)(float));
    float cost();
    float train(unsigned int iterations);
    void initializeParams();
    void test();
    friend class Neuron;


};
void test(NN);
#endif //NNL_HPP





