#include "NNL.hpp"
#ifndef NNL_CPP
#define NNL_CPP
#define DELTA 1e-5
#define RATE 1e-2
namespace{
    void srandf(){
        srand(time(0));
    }
    void srandf(int i){
        srand(i);
    }
    float randf(){
        return (float)rand() / (float)RAND_MAX;
    }
    auto sigmoid = [](float i){
        return 1.f/(1.f+exp(-i));
    };
    auto reLu = [](float i){
        if (i<0){
            i = 0;
        }
        return i; 
    };
};

model::model(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape){
    this->numIns = trainingSet[0].inputs.size();
    this->numOuts = trainingSet[0].outputs.size();
    this->trainingSet = trainingSet;
    this->shape = shape;
    unsigned int numWeights = 0;
    unsigned int numBias = 0;
    unsigned int numLayers = this->shape.size();
    for(unsigned int i = 0; i < numLayers; ++i){
        unsigned int layerSize = shape[i];
        unsigned int prevLayerSize;
        if (i){
            prevLayerSize = shape[i-1];
        }else{
            prevLayerSize = numIns;
        }
        numBias += layerSize;
        numWeights += layerSize * prevLayerSize;
    }//numWeights & numBias now initialized
    this->weightsList = std::vector<float>(numWeights);
    this->biasList = std::vector<float>(numBias); 
    this->numNeurons = numBias;
};
model::model(){
    trainingSet = std::vector<DataPoint>();
    shape = std::vector<unsigned int>();
    weightsList = std::vector<float>();
    biasList = std::vector<float>();
};
//NEURON
unsigned int Neuron::nextId = 0;
Neuron::Neuron(){
    wptr = std::vector<float*>();
};
Neuron::Neuron(unsigned int numInputs, float(*activationFunction)(float), unsigned int& wCursor, model& model){
    id = nextId++;
    activate = activationFunction;
    wptr = std::vector<float*>(numInputs);

    for(unsigned int i = 0; i < numInputs; ++i){
        wptr[i] = &(model.weightsList[wCursor++]);
    }

    b = &(model.biasList[id]);//1 bias per neuron so neuron0 aligns with bias0
};

float Neuron::update(std::vector<float> ins){
    val = 0;
    for(unsigned int i = 0; i < wptr.size(); ++i){
        val += ins[i]* (*wptr[i]);
    }
    val = activate(val + *b);
    return val;
}
//END NEURON

//NN

//members
NN::
NN::NN(struct model& m, float(*activationFunction)(float)){
    this->m = m;
    unsigned int wCursor = 0;
    this->neuronList = std::vector<Neuron>(m.numNeurons);
    for (Neuron& n : this->neuronList){
        n = Neuron(m.numIns, activationFunction, wCursor, m);
    }
};
NN::NN(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape, float(*activationFunction)(float)){
    this->m = model(trainingSet,shape);
    unsigned int wCursor = 0;
    for (Neuron& n : this->neuronList){
        n = Neuron(m.numIns, activationFunction, wCursor, m);
    }
};
std::vector<float> NN::forward(std::vector<float> ins){
    std::vector<float> inputs = ins;
    std::vector<float> outs = std::vector<float>();
    unsigned int numN = this->m.numNeurons;
    unsigned int layer = 0;
    unsigned int nextL = this->m.shape[layer];
    for (unsigned int i = 0; i < numN; ++i){
        Neuron& n = neuronList[i];
        if (i >= nextL){//check for end of layer
            nextL += this->m.shape[++layer];
            ins = outs;
            outs = std::vector<float>();
        }
        outs.push_back(n.update(ins));
    }
    return outs;
};
float NN::cost(){
    float result = 0;
    int items = 0;
    for (unsigned int i = 0; i < this->m.trainingSet.size(); ++i){
        float d = 0;
        DataPoint train = this->m.trainingSet[i];
        std::vector<float> predicted = forward(train.inputs);
        for (int i = 0; i < predicted.size(); ++i){
            d = predicted[i] - train.outputs[i];
            items++;
            result += d * d;
        }
    }
    return result / items;
};
float NN::dCost(float* toMod){
    float delta = DELTA;
    float temp = *toMod;
    float c = cost();
    *toMod += delta;
    float result = (cost() - c)/delta;
    *toMod = temp;
    return result;
};
float NN::gradientDescent(){
    unsigned int numW = m.weightsList.size();
    float deltaSum = 0;
    for(unsigned int i = numW; i > 0; --i){
        float& weightPtr = m.weightsList[i-1];
        float deltaC = dCost(&weightPtr);
        weightPtr -= deltaC*RATE;
        deltaSum+=deltaC;
    }
    return deltaSum;
};
float NN::train(unsigned int iter){
    std::cout<<"Training...\n";
    for (unsigned int i = 0; i < iter; ++i){
        gradientDescent();
        std::cout<<i<<"-cost: "<<cost()<<std::endl;
    }
    std::cout<<"...Training Done!\n";
    return cost();
};
void NN::initializeParams(){
    int w = 0;
    for(int i = 0; i < m.numNeurons; ++i){
        Neuron* n = &neuronList[i]; 
        //std::cout<<n->id<<'\n';
        for (int j = 0; j < n->wptr.size(); ++j){
            neuronList[i].wptr[j] = m.weightsList.data()+w;
            *neuronList[i].wptr[j] = randf();
            //std::cout<<"&w"<<w<<" = "<< m.weightsList.data()+w<<'\t';
            //std::cout<<"wp"<<j<<" = "<< neuronList[i].wptr[j]<<'\n';
            w+=1;
        }
    }
}
void NN::test(){
    for (unsigned int i = 0; i < m.trainingSet.size(); ++i){
        DataPoint train = this->m.trainingSet[i];

        std::cout<<"inputs: {";
        for (float x : train.inputs){
            std::cout<<x<<", ";
        }
        std::cout<<"} -> outputs: {";

        std::vector<float> predicted = forward(train.inputs);
        for (int i = 0; i < predicted.size(); ++i){
            std::cout<<predicted[i]<<", ";
        }
        std::cout<<"}\n";
    }
    return;
}

//END NN
//testMain
int main(void){
    srandf();
    randf();

    


    int iterations = 1500;
    if (iterations < 0){iterations *= -1;}
    std::vector<DataPoint> XOR = {
        {{0,0}, {0}},
        {{1,0}, {1}},
        {{0,1}, {1}},
        {{1,1}, {0}}
    };
    std::vector<dataPoint> SQRT = {
        {{0},{0}},
        {{2},{(float)sqrt(2)}},
        {{4},{2}},
        {{9},{3}},
    };
    std::vector<unsigned int> shape = {8,8,1};
    srandf();
    

    struct model m = model(SQRT, shape);
    NN network = NN(m, reLu);
    network.initializeParams();
    std::cout<<"cost: " << network.cost()<<std::endl;
    
    float cost = network.train(iterations);
    std::cout<<"cost: " << cost <<std::endl;
    network.test();
}

/*----------------------------------------------------------------------
TODO:
--------

fix model/neuron initialization
    - notes:  
    - the weights pointers in the neurons dont point to the weights in the model's list of weights

change first public in NN and Neuron declaration to private: public for debugging only

-----------------------------------------------------------------------*/
#endif //NNL_CPP
