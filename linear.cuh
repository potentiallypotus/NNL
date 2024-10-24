#ifndef LINEAR_CUH
#define LINEAR_CUH

#include <cuda_runtime.h>
class Matrix{	
	float* deviceData;
public:
	unsigned int rows, cols;
	float* hostData;
	
	Matrix(unsigned int rows, unsigned int cols);
	~Matrix();

	void allocate();
	void copyToDevice();
	void copyToHost();
	void freeDevice();

	void add(const Matrix& other);
};

//cuda kernels
__global__ void matrixAddKernel(float* a, float* b, float* result, unsigned int rows, unsigned int cols);

#endif //LINEAR_CUH
