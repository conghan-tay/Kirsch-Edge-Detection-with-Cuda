/*****************************************************************
/*Start Header
/*!
\file main.cpp
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
// CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include <stdint.h>
#include "bmp.h"
#include "edge.h"

#define DEVPROP

const static char *sSDKsample = "[kirsch]\0";

int main(int argc, char **argv)
{
	uchar *h_DataIn;
	uchar *cpuOutput_image;
	uchar *d_DataIn;
	uchar *d_DataOut;
	int *d_Mask;
	bmp_header header;
	StopWatchInterface *hTimer = NULL;
	int PassFailFlag = 1;
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc != 4) {
		printf
		("Usage: kirsch <InFile> <GPUOutFile> <CPUOutFile> \n\n");
		exit(0);
	}

	// set logfile name and start logs
	printf("[%s] - Starting...\n", sSDKsample);

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

#ifdef DEVPROP
	int driverVersion = 0, runtimeVersion = 0;
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10,
		runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("  Total amount of global memory:                 %.2f GBytes (%llu "
		"bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
		(unsigned long long)deviceProp.totalGlobalMem);
	printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
		"GHz)\n", deviceProp.clockRate * 1e-3f,
		deviceProp.clockRate * 1e-6f);
	printf("  Memory Clock rate:                             %.0f Mhz\n",
		deviceProp.memoryClockRate * 1e-3f);
	printf("  Memory Bus Width:                              %d-bit\n",
		deviceProp.memoryBusWidth);

	if (deviceProp.l2CacheSize)
	{
		printf("  L2 Cache Size:                                 %d bytes\n",
			deviceProp.l2CacheSize);
	}

	printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
		"2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
		deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
		deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
		deviceProp.maxTexture3D[2]);
	printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
		"2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
		deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
		deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);
	printf("  Total amount of constant memory:               %lu bytes\n",
		deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n",
		deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n",
		deviceProp.warpSize);
	printf("  Maximum number of threads per multiprocessor:  %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of threads per block:           %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %lu bytes\n",
		deviceProp.memPitch);

#endif
	sdkCreateTimer(&hTimer);

	printf("Initializing data...\n");
	printf("...reading input data\n");
	printf("...allocating CPU memory.\n");

	bmp_read(argv[1], &header, &h_DataIn);

	int imageWidth = header.width;
	int imageHeight = header.height;
	int imageChannels = 3;

	double dAvgSecs;
	uint byteCount = imageWidth*imageHeight*imageChannels*sizeof(unsigned char);

	uchar *gpuOutput_image = (uchar *)malloc(sizeof(uchar) * byteCount);
	memcpy_s(gpuOutput_image, byteCount, h_DataIn, byteCount);

	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_DataIn, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_DataOut, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_Mask, 8 * 9 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_DataIn, h_DataIn, byteCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Mask, kirschFilter, 8 * 9 * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	kirschEdgeDetectorGPU(	d_DataIn, d_Mask, d_DataOut, 
							imageChannels, imageWidth, imageHeight  );

//	printf("\nValidating GPU results...\n");
//	printf(" ...reading back GPU results\n");
	checkCudaErrors(cudaMemcpy(gpuOutput_image, d_DataOut, byteCount, cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);

	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("kirschEdgeDetectionGPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("kirsch edge detection, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1);

	printf("Shutting down GPU...\n\n");

	checkCudaErrors(cudaFree(d_Mask));
	checkCudaErrors(cudaFree(d_DataIn));
	checkCudaErrors(cudaFree(d_DataOut));
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	cpuOutput_image = (uchar *)malloc(sizeof(uchar) * byteCount);
	memcpy_s(cpuOutput_image, byteCount, h_DataIn, byteCount);

	//printf("...kirschEdgeDetectorCPU()\n");
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	kirschEdgeDetectorCPU(
		h_DataIn, (int *) kirschFilter, cpuOutput_image,
		imageChannels, imageWidth, imageHeight
	);
	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); 
	printf("kirschEdgeDetectorCPU() time: %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("kirschEdgeDetectorCPU, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount);
	printf("CPU version done...\n\n");

	sdkDeleteTimer(&hTimer);
#if 1
	printf("...comparing the results of GPU and CPU version\n");
	for (uint i = 0; i < byteCount; i++)
		if (cpuOutput_image[i] != gpuOutput_image[i])
		{
			PassFailFlag = 0;
			break;
		}
	printf(PassFailFlag ? " ...kirsch edge detection matched\n\n" : " ***kirsch edge detection do not match!!!***\n\n");
#endif
	// write data to output bmp image file
	//GPU results
	bmp_write(argv[3], &header, gpuOutput_image);//free inside bmp_write
	//CPU results
	bmp_write(argv[2], &header, cpuOutput_image);
	free(h_DataIn);	

//	printf("%s - Test Summary\n", sSDKsample);
#if 0
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		return -1;
	}

	printf("Test passed\n");
#endif

	return 0;
}

