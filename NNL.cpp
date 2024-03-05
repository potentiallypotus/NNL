#include <cstddef>
#include "NNL.hpp"

#ifndef NNL_CPP
#define NNL_CPP
#define FINITE_DIFF 0
#define EPOCH 1500
#define DELTA 1e-5
#if FINITE_DIFF
#define RATE 1e-2
#else
#define RATE 1e-1
#endif
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
    float sigP (float i){
        float s = sigmoid(i);
        return s * (1-s);
    }

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
    this->weightsList = list(numWeights);
    this->biasList = list(numBias); 
    this->numNeurons = numBias;
};
model::model(){
    trainingSet = std::vector<DataPoint>();
    shape = std::vector<unsigned int>();
    weightsList = list();
    biasList = list();
};
//NEURON
unsigned int Neuron::nextId = 0;
Neuron::Neuron(){
    wptr = std::vector<float*>();
};
Neuron::Neuron(float val){
    wptr = std::vector<float*>();
    this->val = val;
};
Neuron::Neuron(unsigned int numInputs, float(*activationFunction)(float), unsigned int& wCursor, model& model, int layer){
    id = nextId++;
    l = layer;
    activate = activationFunction;
    wptr = std::vector<float*>(numInputs);

    for(unsigned int i = 0; i < numInputs; ++i){
        wptr[i] = &(model.weightsList[wCursor++]);
    }

    b = &(model.biasList[id]);//1 bias per neuron so neuron0 aligns with bias0
};

float Neuron::update(list ins){
    float temp = 0;
    for(unsigned int i = 0; i < wptr.size(); ++i){
        temp += ins[i]* (*wptr[i]);
    }
    z = (temp + *b);
    val = activate(z);
    return val;
}
//END NEURON

//NN

//members
NN::NN(struct model& m, float(*activationFunction)(float)){
    this->m = m;
    this->neuronList = std::vector<Neuron>(m.numNeurons);
    inputLayer = std::vector<Neuron>(m.numIns);
    layerList = std::vector<Neuron*>(m.shape.size());
    expectedOuts = std::vector<float>(m.numOuts);
    unsigned int wCursor = 0;
    int layer = 0;
    int i = 1;
    for (Neuron& n : this->neuronList){
        n = Neuron(m.numIns, activationFunction, wCursor, m, layer);
        if(i == 1)layerList[layer] = &n;
        if(i == m.shape[layer]){
            i = 1;
            layer++;
        }else i++;
    }
};
NN::NN(const std::vector<DataPoint> &trainingSet, const std::vector<unsigned int> &shape, float(*activationFunction)(float)){
    this->m = model(trainingSet,shape);
    inputLayer = std::vector<Neuron>(m.numIns);
    expectedOuts = std::vector<float>(m.numOuts);
    unsigned int wCursor = 0;
    this->neuronList = std::vector<Neuron>(m.numNeurons);
    int layer = 0;
    int i = 1;
    for (Neuron& n : this->neuronList){
        n = Neuron(m.numIns, activationFunction, wCursor, m, layer);
        if(i == m.shape[layer]){
            layerList[layer] = &n;
            i = 1;
            layer++;
        }
    }
};
list NN::forward(list ins, list eOuts){
    expectedOuts = eOuts;
    list inputs = ins;
    list outs = list();
    unsigned int numN = this->m.numNeurons;
    unsigned int layer = 0;
    unsigned int nextL = this->m.shape[layer];
    //static_assert(inputLayer.size() == ins.size());
    for (int i = 0; i < ins.size(); ++i){
        inputLayer[i] = Neuron(ins[i]);
    }
    for (unsigned int i = 0; i < numN; ++i){
        Neuron& n = neuronList[i];
        if (i >= nextL){//check for end of layer
            nextL += this->m.shape[++layer];
            ins = outs;
            outs = list();
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
        list predicted = forward(train.inputs, train.outputs);
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
float NN::backProp(std::vector<matrix>& weightsMatricies, matrix& biasMatrix, matrix& activatedMatrix){
    

    for (int i = 0; i < m.trainingSet.size(); ++i){
        for (int nI = neuronList.size()-1; nI >= 0; --nI){
            // l = current layer index
            // y = expected value of output node
            // la = pointer to the above layer
            // lc = current layer
            // lb = layer below
            

            Neuron* n = &neuronList[nI];
            int l = n->l;
            Neuron* lc = layerList[l];
            float dcda = 0;
            if (l == m.shape.size()-1){//if output layer
                float y = expectedOuts[n-lc];
                dcda = 2*(n->val - y);
                n->dcda = dcda;
                activatedMatrix[l][n-lc] += dcda;
            }else{
                Neuron* la = layerList[l+1];
                int k = n - lc;
                for (int j = 0; j < m.shape[l+1]; ++j){
                dcda += (*(la+j)->wptr[k]) * (sigP(la[j].z)) * (la[j].dcda);
                n->dcda = dcda;
                activatedMatrix[l][n-lc] += dcda;
                }
            }
            float dcdbj = sigP(n->z) * dcda;
            biasMatrix[l][n-lc] -= rate * dcdbj;
            Neuron* lb = layerList[l-1];
            for (int k = 0; k < n->wptr.size(); ++k){
                if (l == 0){
                    float dcdwjk = inputLayer[k].val * sigP(n->z) * dcda;
                    weightsMatricies[l][n-lc][k] -= rate * dcdwjk;
                }else{
                    float dcdwjk = lb[k].val * sigP(n->z) * dcda;
                    weightsMatricies[l][n-lc][k] -= rate * dcdwjk;
                }
            }
        }
    }
}
float NN::gradientDescent(){
#if FINITE_DIFF
    unsigned int numW = m.weightsList.size();
    float deltaSum = 0;
    for(unsigned int i = numW; i > 0; --i){
        float& weightPtr = m.weightsList[i-1];
        float deltaC = dCost(&weightPtr);
        weightPtr -= deltaC*RATE;
        deltaSum+=deltaC;
    }
    return deltaSum;
#else
    // for ( int i = 0; i < m.trainingSet.size(); ++i){
    //     DataPoint train = m.trainingSet[i];
    //     forward(train.inputs, train.outputs);
    //     backProp();
    // }
    srandf();
    std::vector<matrix> wMats = std::vector<matrix>(layerList.size());
    int L = 0;
    for (matrix& wmat : wMats){
        wmat = matrix(m.shape[L]);
        for (list& js : wMat){
            js= list(L ? m.shape[L-1] : m.numIns);
            for (float& ks : js){
                ks = 0;
            }
        }
        L++;
    }
    matrix bMat = matrix(m.numNeurons);
    L = 0;
    for (list& Layer: bMat){
        Layer = list(m.shape[L]);
        for (float& B : Layer){
            b = 0;
        }
        L++;
    }
    matrix aMat = matrix(m.numNeurons);
    L = 0;
    for (list& Layer: bMat){
        Layer = list(m.shape[L]);
        for (float& B : Layer){
            b = 0;
        }
        L++;
    }
    for (int i = 0; i < m.trainingSet; ++i){
        dataPoint train = m.trainingSet[i];
        forward(train.inputs, train.outputs);
        backProp();
    }



#endif
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

        list predicted = forward(train.inputs, train.outputs);
        for (int i = 0; i < predicted.size(); ++i){
            std::cout<<predicted[i]<<", ";
        }
        std::cout<<"}\n";
    }
    return;
}

//END NN
//testMain
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
int main(void){
    srandf();
    randf();

    


    int iterations = EPOCH;
    if (iterations < 0){iterations *= -1;}
    std::vector<DataPoint> XOR = {
        {{0,0}, {0}},
        {{1,0}, {1}},
        {{0,1}, {1}},
        {{1,1}, {0}}
    };
    std::vector<DataPoint> SQRT = {
        {{0},{0}},
        {{2},{(float)sqrt(2)}},
        {{4},{2}},
        {{9},{3}},
    };
    std::vector<DataPoint> BIN = buildBinSet();

    std::vector<unsigned int> shape = {2,1};
    srandf();
    

    struct model m = model(XOR, shape);
    NN network = NN(m, sigmoid);
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


make the neuron list in NN a 2d array, such that it can be indexed by layer then number
    OR
give NN a list of pointers to the beginning of each layer
change first public in NN and Neuron declaration to private: public for debugging only

-----------------------------------------------------------------------*/
#endif //NNL_CPP
