#include "NNL.hpp"
#include <new>
#include <iostream>
#include <cassert>

Network::Network(unsigned int numberOfLayers, unsigned int *shape) {
    numLayers = numberOfLayers;
    this->shape = shape;
    void* newMem = operator new[](numLayers * sizeof(Layer));
    pointerToLayers = static_cast<Layer*>(newMem);
    new (&pointerToLayers[0]) Layer(shape[0]);
    for (unsigned int i = 1; i < numLayers; i++){
        new (&pointerToLayers[i]) Layer(shape[i], shape[i-1]);
    }
}
Network::~Network(){
    if (shape){
        delete[] shape;
    }
    if (pointerToLayers){
        for (int i = 0; i < numLayers; i++){
            pointerToLayers[i].~Layer();
        }
        operator delete[](pointerToLayers);
    }
}
void Network::forward(){
    for (int i = 1; i < numLayers; i++){
        pointerToLayers[i].forward(pointerToLayers[i-1].valueArray);
    }
}

Layer::Layer(unsigned int layerSize, unsigned int prevLayerSize) :
    layerSize(layerSize), 
    previousLayerSize(prevLayerSize),
    weights(layerSize, previousLayerSize),
    valueArray(layerSize, 1),
    biasArray(layerSize, 1)
{
    for (unsigned int i = 0; i < layerSize* previousLayerSize; i++){
        weights.hostData[i] = 1.0f;
    }
    for (unsigned int i = 0; i < layerSize; i++){
        biasArray.hostData[i] = 0.0f;
    }
}
Layer::Layer(unsigned int layerSize): layerSize(layerSize), valueArray(layerSize, 1){
}
Layer::~Layer(){
}
void Layer::forward(Matrix& prevLayerVals){
    this->weights.mult(prevLayerVals, valueArray);
    this->valueArray.add(biasArray);
    
}
