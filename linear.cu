#include "linear.cuh"
#include <cassert>
#include <iostream>
#include <iomanip>
Matrix::Matrix(unsigned int r, unsigned int c) {
	rows = r;
	cols = c;
	hostData = new float[rows * cols];  // Allocate on host
	deviceData = nullptr;
}
Matrix::Matrix(){
    rows = 0; cols = 0;
    hostData = nullptr;
    deviceData = nullptr;
}

Matrix::~Matrix() {
    if (hostData) delete[] hostData;                   // Free host memory
    freeDevice();                    // Free device memory
}

void Matrix::allocate() {
    if (deviceData == nullptr){ 
	cudaError_t err = cudaMalloc(&deviceData, rows * cols * sizeof(float)); // Allocate on device
	if ( err != cudaSuccess){
	    std::cout<<"error in cudaMalloc: "<< cudaGetErrorString(err)<<std::endl;
	}
    }
}

void Matrix::copyToDevice() {
    if(!deviceData) {
	std::cout<< "attempt to write to unallocated pointer!\n";
	return;
    }
    cudaError_t err = cudaMemcpy(deviceData, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    if ( err != cudaSuccess){
	std::cout<<"error in cudaMemcpy from function Matrix::copyToDevice(): " << cudaGetErrorString(err) <<std::endl;
    }
}

void Matrix::copyToHost() {
    if(!deviceData) {
	std::cout<< "attempt to read from unallocated pointer!\n";
	return;
    }
    cudaError_t err = cudaMemcpy(hostData, deviceData, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess){
	std::cout<<"error in cudaMemcpy from function Matrix::copyToHost(): " << cudaGetErrorString(err) <<std::endl;
    }
}

void Matrix::freeDevice() {
    if (deviceData) {
        cudaFree(deviceData);
        deviceData = nullptr;
    }
}
void Matrix::print(){
    std::cout<<"{";
    for (int i = 0; i < rows; i++){
        if (i != 0) std::cout<<" ";
        for (int j = 0; j < cols; j++){
            std::cout<<" ";
	    std::cout<<std::setw(4)<< hostData[i*cols + j];
            if (!(i == rows-1 && j == cols-1))
                std::cout<<",";
            else
                std::cout<<" }";
        }
            std::cout<<"\n";
    }
}

void Matrix::add(Matrix& other){
	other.allocate();
	other.copyToDevice();
	unsigned int blockSize = 128;
	unsigned int totalElements = rows * cols;
	unsigned int numBlocks = (totalElements + blockSize - 1) / blockSize;
	matrixAddKernel<<<numBlocks, blockSize>>>(deviceData, other.deviceData, rows, cols);
	cudaDeviceSynchronize();
	copyToHost();
}
#define BLOCKDIM 3
void Matrix::mult(Matrix& other, Matrix& result){
    assert(this->rows == result.rows);
    assert(other.cols == result.cols);
    assert(this->cols == other.rows);
    unsigned int m, n, p;
    m = result.rows;
    n = other.rows;
    p = result.cols;
    this->allocate();
    this->copyToDevice();
    other.allocate();
    other.copyToDevice();
    result.allocate();
#define Naive 
#ifdef Naive
    assert(p < 1024);
    cudaGetLastError();
    dim3 gridDim(m);
    dim3 blockDim(p);
    naiveMult<<<gridDim, blockDim>>>(this->deviceData, other.deviceData, result.deviceData, m, n, p);
#endif
#ifndef Naive
    //how many threads per one dimension of a square block, threads per block is this number squared
    int blockDimension = 16;//if this number is changed, the SHMEM_SIZE must change to reflect it
    //x dimension of the grid layout, 
    int gridDimx = (result.cols + blockDimension - 1) / blockDimension;
    int gridDimy = (result.rows + blockDimension - 1) / blockDimension;
    dim3 blockSize(blockDimension, blockDimension);
    dim3 gridSize(gridDimx, gridDimy);
    cudaGetLastError();
    matrixMultKernel<<<gridSize, blockSize>>>(deviceData, other.deviceData, result.deviceData, m, n, p, blockDimension);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    std::cout<<"Error in Kernel Launch: "<<cudaGetErrorString(err)<<std::endl;
    cudaDeviceSynchronize();
    result.copyToHost();
}

//Cuda kernels

__global__ void naiveMult(float* a, float* b, float* result, unsigned int m, unsigned int n, unsigned int p){
    int col = threadIdx.x;
    int row = blockIdx.x;
    float partialSum = 0;
    for (int i = 0; i  < n; i++){
	if(row < m && col < n)
	   partialSum += a[row * n + i] * b[col + i * p];
    }
    if(row < m && col < p){
	result[row*p + col] = partialSum;
    }
}
__global__ void matrixAddKernel(float* a, float* b, unsigned int rows , unsigned int cols){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < rows*cols){
		a[idx] += b[idx];
	}
}

#define SHMEM_SIZE BLOCKDIM * BLOCKDIM * sizeof(float)
__global__ void matrixMultKernel(float* a, float* b, float* result,
				 unsigned int m, unsigned int n,
				 unsigned int p, unsigned int tileSize){
    __shared__ float A[SHMEM_SIZE];
    __shared__ float B[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //calculate global row and collumn positions for this thread
    int row = by * tileSize + ty;
    int col = bx * tileSize + tx;


    float partialSum = 0;
    //sweep over tiles in the matrix
    for (int i = 0; i < (n / tileSize); i++){
	//for each tile in the a and b matrices, we want to populate the shared memory tile by having each thread in the block populate one element.
	 //for this reason we want to ensure that the tile size is equal to the block size ie SHMEM_size = sizeof(float) * bblockdimx.x * blockdimx.y
	if ((row < m) && ((i * tileSize + tx) < n)){
	    A[ty * tileSize + tx] = a[row * n + (i * tileSize + tx)];
	}else {
	    A[ty * tileSize + tx] = 0.0f;
	}
	if (((i * tileSize + ty) < n) && (col < p)){
	    B[ty * tileSize + tx] = b[(i * tileSize + ty) * p + col];
	}else {
	    B[ty * tileSize + tx] = 0.0f;
	}
	__syncthreads();

	for(int j = 0; j < tileSize; j++){
	    partialSum += A[ty * tileSize + j] * B[j * tileSize + tx];
	}

	__syncthreads();
    }
    if ((row < m) && (col < p)){
	result[row*p + col] = partialSum;
    }
}
