#include "linear.cuh"

Matrix::Matrix(unsigned int r, unsigned int c) {
	rows = r;
	cols = c;
	hostData = new float[rows * cols];  // Allocate on host
	deviceData = nullptr;
}

Matrix::~Matrix() {
    delete[] hostData;                   // Free host memory
    freeDevice();                    // Free device memory
}

void Matrix::allocate() {
    cudaMalloc(&deviceData, rows * cols * sizeof(float)); // Allocate on device
}

void Matrix::copyToDevice() {
    cudaMemcpy(deviceData, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::copyToHost() {
    cudaMemcpy(hostData, deviceData, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::freeDevice() {
    if (deviceData) {
        cudaFree(deviceData);
        deviceData = nullptr;
    }
}

void Matrix::add(const Matrix& other){
	float* result;
	cudaMalloc(&result, rows * cols * sizeof(float));
	unsigned int blockSize = 128;
	unsigned int totalElements = rows * cols;
	unsigned int numBlocks = (totalElements + blockSize - 1) / blockSize;
	matrixAddKernel<<<numBlocks, blockSize>>>(deviceData, other.deviceData, result, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(deviceData, result, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
	copyToHost();
}

//Cuda kernels
__global__ void matrixAddKernel(float* a, float* b, float* result, unsigned int rows , unsigned int cols){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < rows*cols){
		result[idx] = a[idx] + b[idx];
	}
}
