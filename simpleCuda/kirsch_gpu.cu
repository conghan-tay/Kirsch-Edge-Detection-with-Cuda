/*****************************************************************
/*Start Header
/*!
\file kirsch_gpu.cu
\author Tay Conghan, conghan.tay, 390007115
\par email: conghan.tay\@digipen.edu
\date Oct 6, 2018
\brief
Copyright (C) 2018 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#include <helper_cuda.h>
#include <algorithm>
#include "edge.h"

#define TILE_WIDTH 16
//#define O_TILE_WIDTH 14
#define O_TILE_WIDTH 14
#define MASK_WIDTH 3
const int BLOCK_WIDTH = O_TILE_WIDTH + MASK_WIDTH - 1;

const int SM_WIDTH = TILE_WIDTH + MASK_WIDTH - 1;
const int WIDTH_SQUARE = TILE_WIDTH * TILE_WIDTH;
const int NUM_LOADS = (( ((float)SM_WIDTH*SM_WIDTH) / (float)WIDTH_SQUARE) + 1.f);
const int DIVIDE_WIDTH = TILE_WIDTH + 2;

//@@ INSERT CODE HERE
//Use of const  __restrict__ qualifiers for the mask parameter 
//informs the compiler that it is eligible for constant caching

///Design 1: The size of each thread block matches the size of an 
///output tile. All threads participate in calculating output elements.

// BLOCK_WIDTH will be SHARED MEMORY SIZE
// TILE_WIDTH will be Block size
__global__ void convolution(unsigned char *I, 
							const int *__restrict__ M,
							unsigned char *P, 
							int channels, 
							int width,
							int height) 
{
	__shared__ int Ns[SM_WIDTH][SM_WIDTH][3];

	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	int size = width * height;

	for (int i = 0; i < NUM_LOADS; ++i)
	{
		int A = ty * TILE_WIDTH + tx + (i*(WIDTH_SQUARE));
		int sy = A / DIVIDE_WIDTH;
		int sx = A%DIVIDE_WIDTH;

		int iY = (by * TILE_WIDTH) + sy - 1;
		int iX = (bx * TILE_WIDTH) + sx - 1;

		if (sy < SM_WIDTH && sx < SM_WIDTH)
		{
			if (iY >= 0 && iY < height && iX >= 0 && iX < width)
			{
				Ns[sy][sx][0] = I[iY * width + iX];
				Ns[sy][sx][1] = I[iY * width + iX + size];
				Ns[sy][sx][2] = I[iY * width + iX + 2 * size];
			}
			else
			{
				Ns[sy][sx][0] = 0;
				Ns[sy][sx][1] = 0;
				Ns[sy][sx][2] = 0;
			}
		}
	}

	__syncthreads();

	int row_o = by * TILE_WIDTH + ty;
	int col_o = bx * TILE_WIDTH + tx;

	if (row_o < height && col_o < width)
	{
		int max_sum = 0;
		int max_sum1 = 0;
		int max_sum2 = 0;
		for (int m = 0; m < 8; ++m)
		{
			int sum = 0;
			int sum1 = 0;
			int sum2 = 0;
			for (int i = 0; i < MASK_WIDTH; ++i)
			{
				for (int j = 0; j < MASK_WIDTH; ++j)
				{
					sum += *(M + m * 9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx][0];
					sum1 += *(M + m * 9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx][1];
					sum2 += *(M + m * 9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx][2];
				}
			}

			max_sum = sum > max_sum ? sum : max_sum;
			max_sum1 = sum1 > max_sum1 ? sum1 : max_sum1;
			max_sum2 = sum2 > max_sum2 ? sum2: max_sum2;
		}

		P[(row_o * width + col_o)] = mymin(mymax(max_sum / 8, 0), 255);
		P[(row_o * width + col_o) + size] = mymin(mymax(max_sum1 / 8, 0), 255);
		P[(row_o * width + col_o) + 2 * size] = mymin(mymax(max_sum2 / 8, 0), 255);

	}
	__syncthreads();
}

///Design 2: The size of each thread block matches the size of 
///an input tile. Each thread loads one input element into the 
///shared memory.

__global__ void convolution2(unsigned char *I,
							const int *__restrict__ M,
							unsigned char *P,
							int channels,
							int width,
							int height)
{
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	
	int tx = threadIdx.x; int ty = threadIdx.y;
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;

	int row_i = row_o - 1;
	int col_i = col_o - 1;

	int size = width * height;

	

	for (int layer = 0; layer < channels; ++layer) {
		int max_sum = 0;
		if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
		{
			Ns[ty][tx] = I[row_i * width + col_i + layer * size];
		}
		else
		{
			Ns[ty][tx] = 0.f;
		}

		__syncthreads();
		if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
		{
			for (int m = 0; m < 8; ++m)
			{
				int sum = 0;
				for (int i = 0; i < MASK_WIDTH; ++i)
				{
					for (int j = 0; j < MASK_WIDTH; ++j)
					{
						sum += *(M + m*9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx];
					}
				}

				max_sum = sum > max_sum ? sum : max_sum;
			}
			if (row_o < height && col_o < width)
			{
				P[(row_o * width + col_o) + layer * size] = mymin(mymax(max_sum / 8, 0), 255);
			}

		}
		__syncthreads();
	}

	
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU 
////////////////////////////////////////////////////////////////////////////////

extern "C" void kirschEdgeDetectorGPU(
	void *d_ImgDataIn,
	void *d_ImgMaskData,
	void *d_ImgDataOut,
	unsigned imgChannels,
	unsigned imgWidth,
	unsigned imgHeight
)
{
#if 1
	//each channel use blockIdx.z dimension 

	/*dim3 dimGrid((imgWidth - 1) / TILE_WIDTH + 1,
		(imgHeight - 1) / TILE_WIDTH + 1, 3);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);*/
	
	dim3 dimGrid((imgWidth - 1) / TILE_WIDTH + 1,
		(imgHeight - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	convolution << <dimGrid, dimBlock >> >((unsigned char *)d_ImgDataIn,
		(int *)d_ImgMaskData,
		(unsigned char *)d_ImgDataOut,
		(int)imgChannels,
		(int)imgWidth,
		(int)imgHeight);
#else
	//each channel use blockIdx.z dimension 
/*
	dim3 dimGrid((imgWidth-1) / O_TILE_WIDTH + 1,
	(imgHeight-1) / O_TILE_WIDTH + 1, 3);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
*/
	dim3 dimGrid((imgWidth-1) / O_TILE_WIDTH + 1,
		(imgHeight-1) / O_TILE_WIDTH + 1, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	convolution2 << <dimGrid, dimBlock >> >((unsigned char *)d_ImgDataIn,
		(int *)d_ImgMaskData,
		(unsigned char *)d_ImgDataOut,
		(int)imgChannels,
		(int)imgWidth,
		(int)imgHeight);
#endif
	getLastCudaError("Compute the kirsch edge detection failed\n");
	cudaDeviceSynchronize();
}

