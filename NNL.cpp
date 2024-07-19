#include <cstddef>
#include "NNL.hpp"


#ifndef NNL_CPP
#define NNL_CPP


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

    //activations
    double linear_actF(double i){return i;}

    double sigmoid_actF(double i) {return (double)(1.f/(1.f+exp(-i)));}

    double reLu_actF(double i){
        if (i<0) i = 0;
        return (double)i; 
    }
    double binary_actF(double i) {return i>0;}
    //derivative of the activation function
    double linearPrime(double i) {return 1;}
    double sigmoidPrime(double i){return sigmoid_actF(i) * (1-sigmoid_actF(i));}
    double reLuPrime(double i){ return i>0;}
    double binaryPrime(double i) {return 0;}
    double sec2h(double i) {return 1.f/cosh(i)/cosh(i);}

    
    
};
// derivative functions are found at + 1 the activation function
const actF NN::activationFunction[] = {
    &sigmoid_actF,  &sigmoidPrime,  
    &reLu_actF,     &reLuPrime, 
    &binary_actF,   &binaryPrime,
    &linear_actF,   &linearPrime,
    &tanh,  &sec2h
}; 
model::model(const std::vector<DataPoint> trainingSet, const std::vector<Uint> shape){
    this->trainingSet = trainingSet;
    this->shape = shape;
    numLayers = shape.size();
    numIns = trainingSet[0].inputs.size();
    numOuts = trainingSet[0].outputs.size();
    Uint numneurons = 0;
    for (Uint layersize : shape){
        numneurons += layersize;
    }
    weights = matrix3(numLayers);
    int L = 0;
    for (matrix& layer : weights){
        layer = matrix(shape[L]);
        for (list& js : layer){
            js= list(L ? shape[L-1] : numIns);
            for (double& ks : js){
                ks = 0;
            }
        }
        L++;
    }
    biases = matrix(numneurons);
    L = 0;
    for (list& layer: biases){
        layer = list(shape[L]);
        for (double& B : layer){
            B = 0;
        }
        L++;
    }
};
model::model(){
    trainingSet = std::vector<DataPoint>();
    shape = std::vector<Uint>();
    weights = matrix3();
    biases = matrix();

};


//NN

//members
NN::NN(struct model& m, double rate)
:
AFI(m.numLayers),
a(m.numLayers),
z(m.numLayers),
expectedOuts(m.numOuts),
weightDelta(m.numLayers),
bDelta(m.numLayers),
dCda(m.numLayers)
{
    this->rate = rate;
    this->m = m;
    numIns = m.trainingSet[0].inputs.size();
    numOuts = m.trainingSet[0].outputs.size();
    numLayers = m.shape.size();
    
    numNeurons = 0;
    for (Uint layersize : m.shape){
        numNeurons += layersize;
    }

    for (Uint l = 0; l < numLayers; ++l){
        AFI[l] = sig;
        a[l] = list(m.shape[l]);
        z[l] = list(m.shape[l]);
        bDelta[l] = list(m.shape[l]);
        dCda[l] = list(m.shape[l]);
        weightDelta[l] = matrix(m.shape[l]);
        for (Uint j = 0; j < m.shape[l]; ++j){
            weightDelta[l][j] = list(l? m.shape[l-1] : m.numIns);
        }
    }
    resetTemps();
    
};
list NN::forward(list ins, list eOuts){
    expectedOuts = eOuts;
    list outs = list();
    
    for (unsigned int l = 0; l < numLayers-1; ++l){
        for (Uint j = 0; j < m.shape[0]; j++){
            actF activate = activationFunction[AFI[l]];
            double sum = 0;
            for (Uint k = 0; k < numIns; k++){
                if(!l){
                    sum+= ins[k] * m.weights[0][j][k];
                }else{
                    sum += a[l-1][k] * m.weights[l][j][k];
                }
            }
            z[l][j] = sum + m.biases[l][j];
            a[l][j] = activate(z[l][j]);
        }
            //std::cout<<"debug3 Layer: "<< l <<"\n";
    }
    
    for (Uint j = 0; j < numOuts; ++j){
        outs.push_back(a[numLayers][j]);
    }
    return outs;
};
float NN::cost(){
    float result = 0;
    int items = 0;
    for (unsigned int i = 0; i < m.trainingSet.size(); ++i){
        float d = 0;
        DataPoint train = this->m.trainingSet[i];
        //std::cout<<"debug2: "<< i <<"\n";
        list predicted = forward(train.inputs, train.outputs);
        for (int i = 0; i < numOuts; ++i){
            d = predicted[i] - train.outputs[i];
            items++;
            result += d * d;
        }
    }
    return result / items;
};
void NN::backProp(){
    auto dCda0 = [this](Uint l, Uint j){
        dCda[l][j] = (a[l][j] - expectedOuts[j]);
        return dCda[l][j];
    };
    auto dCdaL = [this](Uint l, Uint k){
        float result = 0;
        actF dA = activationFunction[AFI[l+1]+1];
        for(int j = 0; j < m.shape[l+1]; j++){
            result += m.weights[l+1][j][k] * dA(z[l+1][j]) * dCda[l+1][j];
        }
        dCda[l][k] = result;
        return result;
    };
    for (int l = m.numLayers-1; l>=0; l--){
        actF dA = activationFunction[AFI[l]+1];
        for(int j = m.shape[l]-1; j >= 0; j--){
            if(l==m.numLayers-1){
                m.biases[l][j] += dA(z[l][j]) * dCda0(l, j);
            }else{
                m.biases[l][j] += dA(z[l][j]) * dCdaL(l, j);
            }
            for(int k = m.shape[l-1]-1; k >= 0; k--){
                m.weights[l][j][k] += a[l][j] * dA(z[l][j]) * dCda[l][j];
            }
        }
    }
}
void NN::gradientDescent(){
    // dataSet trainingSet = dataSet();
    // for ( int i = 0; i < 4; ++i){
    //     trainingSet.push_back(m.trainingSet[randf()*m.trainingSet.size()]);
    // }
    for (int i = 0; i < m.trainingSet.size(); ++i){
        dataPoint train = m.trainingSet[i];
        forward(train.inputs, train.outputs);
        backProp();
    }

    //average
    Uint n = m.trainingSet.size();
    for (Uint l = 0; l < numLayers; ++l){
        for (Uint j = 0; j < m.shape[l]; ++j){
            Uint kmax = l ? m.shape[l-1] : numIns;
            bDelta[l][j] /= n;
            m.biases[l][j] -= bDelta[l][j]*rate;
            for(int k = 0; k < kmax; ++k){
                weightDelta[l][j][k] /= n;
                m.weights[l][j][k] -= weightDelta[l][j][k]* rate;
            }
        }
    }
    resetTemps();
};
void NN::train(Uint iter){
    std::cout<<"Training...\n";
    for (Uint i = 0; i < iter; ++i){
        gradientDescent();
        //std::cout<<i<<"-cost: "<<cost()<<std::endl;
    }
    std::cout<<"...Training Done!\n";
    return;
};
void NN::resetTemps(){
    for(Uint l = 0; l < m.numLayers; ++l){
        for (Uint j = 0; j < m.shape[l]; ++j){
            bDelta[l][j] = 0;
            for (Uint k = 0; k< m.shape[l-1]; ++k){
                weightDelta[l][j][k] = 0;
            }
        }
    }
}
void NN::printState(){
    int N = 0;
    for (int i = 0; i < numLayers; ++i){
        std::cout<<"layer "<< i << ":\n";
        for (int j = 0; j < m.shape[i]; ++j){
            std::cout<<"\tNeuron "<< j << ": "<< z[i][j] << "\n";
            if (i){
                for (int k = 0; k < m.weights[i][j].size(); k++){
                std::cout<<"\t\tweight "<< k <<": "<< m.weights[i][j][k]<< "\n";
                }
            }
        }
    }
}
void NN::test(){
    for (unsigned int i = 0; i < m.trainingSet.size(); ++i){
        DataPoint train = this->m.trainingSet[i];

        std::cout<<"inputs: {";
        for (double x : train.inputs){
            std::cout<<x<<", ";
        }
        std::cout<<"} -> outputs: {";

        list predicted = forward(train.inputs, train.outputs);
        for (int i = 0; i < predicted.size()-1; ++i){
            std::cout<<predicted[i]<<", ";
        }

        std::cout<< predicted[predicted.size()-1]<<"}\n";
    }
    return;
}


/*----------------------------------------------------------------------
TODO:
--------

-----------------------------------------------------------------------*/
#endif //NNL_CPP
