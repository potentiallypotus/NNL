#ifndef LINEAR_CUH
#define LINEAR_CUH

#include <cuda_runtime.h>
class Matrix{	
	float* deviceData;
public:
	unsigned int rows, cols;
	float* hostData;
	
	Matrix(unsigned int rows, unsigned int cols);
	Matrix();
	~Matrix();

	void allocate();
	void copyToDevice();
	void copyToHost();
	void freeDevice();
	void print();

	void add(Matrix& other);
	void mult(Matrix& other, Matrix& result);
};

//cuda kernels
__global__ void matrixAddKernel(float* a, float* b, unsigned int rows, unsigned int cols);
__global__ void matrixMultKernel(float* a, float* b, float* result,
				 unsigned int m, unsigned int n, 
				 unsigned int p, unsigned int tileSize);
__global__ void naiveMult(float* a, float* b, float* result, unsigned int m, unsigned int n, unsigned int p);

#endif //LINEAR_CUH
