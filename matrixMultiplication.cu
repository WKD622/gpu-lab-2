// Matrix multiplication by parts
// Elements stored in row-major order

using namespace std;
#include <stdio.h>
#include <iostream>
#include <fstream>
#define BLOCK_SIZE 16

#include "helper_functions.h"

typedef struct
{	int width;
	int height;
	float *elements;
} Matrix;

// Forward declaration of matrix mult
__global__ void MatMulKernel (const Matrix, const Matrix, Matrix);

// Host code
void MatMul(const Matrix A, const Matrix B, Matrix C, int block_size)
{
	// Load matrices A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**) &d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**) &d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// allocate C in device
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = d_C.width * d_C.height * sizeof(float);
	cudaMalloc((void**) &d_C.elements, size);

	// call kernel
	dim3 dimBlock(block_size, block_size, 1); // define the block size (what is the best value?)
	int gx = ( B.width % dimBlock.x == 0 ) ? (B.width / dimBlock.x) : (B.width / dimBlock.x + 1);
	int gy = ( A.height % dimBlock.y == 0 ) ? (A.height / dimBlock.y) : (A.height / dimBlock.y + 1);
	dim3 dimGrid(gx, gy, 1); //  choose grid size depending on problem size

	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// copy C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= A.height || col >= B.width)
	{
		return;
	}

	for(int e = 0; e < A.width; ++e)
	{
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	}

	C.elements[row * C.width + col] = Cvalue;
}

void MatMulCpu(Matrix A, Matrix B, Matrix C)
{
	if (A.width != B.height)
	{
		return;
	}

	for (int aY = 0; aY < A.width; ++aY)
    {
        for (int bX = 0; bX < B.height; ++bX)
        {
            int out = 0;
			for (int aXbY = 0; aXbY < B.width; ++aXbY)
			{
				out += A.elements[A.width * aXbY + aY] * A.elements[B.width * bX + aXbY];
			}
			C.elements[B.width * bX + aY] = out;
		}
	}
}

int main(int argc, char * const argv[])
{

	int Width = 16;
	int customWidth = 16;

	if(argc < 4)
	{
		std::cout << "Not enough arguments" << std::endl;
		return 1;
	}

	std::string method(argv[1]);

	int cpu = 0;
	int gpu = 0;

	if (method.compare("cpu") == 0)
	{
		cpu = 1;
		std::cout << "cpu,";
	}
	else if(method.compare("gpu") == 0)
	{
		gpu = 1;
		std::cout << "gpu,";
	}

	customWidth = atoi(argv[2]);

	if (customWidth < Width || customWidth % Width != 0)
	{
		std::cout << "Custom width has to be multiply of " << Width << std::endl;
		return 1;
	}

	int block_size = atoi(argv[3]);

	Matrix A;
	Matrix B;
	Matrix C;

	A.width = customWidth;
	B.width = customWidth;
	C.width = customWidth;

	A.height = customWidth;
	B.height = customWidth;
	C.height = customWidth;

	A.elements = new float[customWidth*customWidth];
	B.elements = new float[customWidth*customWidth];
	C.elements = new float[customWidth*customWidth];

	//fill matrices
	std::ifstream A_input;
	std::ifstream B_input;
	A_input.open("A.txt");
	B_input.open("B.txt");

	float a, b;
	A_input >> a;
	B_input >> b;
	int i = 0;
	while (!A_input.eof())
	{	A.elements[i] = a;
		B.elements[i] = b;
		A_input >> a;
		B_input >> b;
		i += 1;
	}
	A_input.close();
	B_input.close();

	if (customWidth != Width)
	{
		const int bound = customWidth * customWidth;
		for (int i = Width; i < bound; ++i)
		{
			A.elements[i] = i % Width;
			B.elements[i] = i % Width;
		}
	}

	std::cout << customWidth << ',';

	if (gpu) {
		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);

		MatMul(A, B, C, block_size);

		sdkStopTimer(&timer);
		float t = sdkGetTimerValue(&timer);
		sdkDeleteTimer(&timer);

		std::cout << t << std::endl;
	}

	std::ofstream C_output;
	C_output.open("C.txt");
	for (int i=0; i<Width; i++)
	{	for (int j=0; j<Width; j++)
			C_output<<C.elements[i*Width+j]<<"\t";
		C_output<<endl;
	}

	if (cpu) {
		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);

		MatMulCpu(A, B, C);

		sdkStopTimer(&timer);
		float tcpu = sdkGetTimerValue(&timer);
		sdkDeleteTimer(&timer);

		std::cout << tcpu << std::endl;

	}

	return 0;
}
