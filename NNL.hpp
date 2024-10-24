#include <vector>
#include "linear.cuh"

class Layer{
    unsigned int layerSize;
    unsigned int previousLayerSize;
    float * valueArray;
    Matrix weights;
};
 
class Network{
    unsigned int numLayers;
    unsigned int* shape;
    Layer* pointerToLayers;
};
