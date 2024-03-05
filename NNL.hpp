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

typedef std::vector<float> list;
typedef std::vector<list> matrix;
typedef struct dataPoint{
    list inputs;
    list outputs;
} DataPoint;
struct model{
    unsigned int numIns;
    unsigned int numOuts;
    unsigned int numNeurons;
    std::vector<DataPoint> trainingSet;
    std::vector<unsigned int> shape;
    list weightsList;
    list biasList;
    
    model(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape);
    model();
};
typedef std::vector<DataPoint> dataSet;
class Neuron{
    public:
    static unsigned int nextId;
    unsigned int id;
    std::vector<float*> wptr;
    float* b;
    float val;
    float z; // the sum(ai*wi) +b
    int l; // the layer the neuron is on
    float(*activate)(float);

    float dcda;

    public:
    Neuron();
    Neuron(float val);
    Neuron(unsigned int numInputs, float(*activationFunction)(float), unsigned int& wCursor, model& model, int layer);
    float update(list neuronInputs);
    friend class NN;


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
    float backProp(std::vector<matrix>& weightsMatricies, matrix& biasMatrix, matrix& activatedMatrix);
    friend class Neuron;


};
void test(NN);
#endif //NNL_HPP





