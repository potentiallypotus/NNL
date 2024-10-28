#include <vector>
#include "linear.cuh"

class Layer{
public:
    unsigned int layerSize;
    unsigned int previousLayerSize;
    Matrix valueArray;
    Matrix biasArray;
    Matrix weights;

    Layer(unsigned int layerSize, unsigned int prevLayerSize);
    Layer(unsigned int layerSize);
    ~Layer();
    void forward(Matrix& prevLayerVals);
};
 
class Network{
public:
    unsigned int numLayers;
    unsigned int* shape;
    Layer* pointerToLayers;
    Network(unsigned int numberOfLayers, unsigned int* shape);
    ~Network();
    void forward();
};
