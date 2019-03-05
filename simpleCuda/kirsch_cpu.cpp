/*****************************************************************
/*Start Header
/*!
\file kirsch_cpu.cu
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
#include "edge.h"
#define Mask_width 3
#define Mask_radius Mask_width/2
//#define COLOR_INTERLEAVE
extern "C" void kirschEdgeDetectorCPU(
	const unsigned char *data_in,
	const int *mask,
	unsigned char *data_out,
	const unsigned channels,
	const unsigned width,
	const unsigned height
	)
{
	unsigned int size, i, j, layer;

	size = height * width;

	for (layer = 0; layer < channels; layer++) {
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				int max_sum;
				max_sum = 0;
				/*
				Perform convolutions for
				all 8 masks in succession. Compare and find the one
				that has the highest value. The one with the
				highest value is stored into the final bitmap.
				*/
				for (unsigned m = 0; m < 8; ++m) {
					int sum;
					int sum1, sum2, sum3;
					sum = 0;
					/* Convolution part */
					if (i >= 1 && i + 1 < height && j >= 1 && j+1 < width) {
						//no ghost cell
						for (int k = -1; k < 2; k++) {
							for (int l = -1; l < 2; l++) {
#ifdef COLOR_INTERLEAVE
								sum =
									sum + *(mask + m * 9 + (k + 1)*channels + (l + 1))*
									(int)data_in[((i + k) * width + (j + l))*channels + layer];
#else
								sum =
									sum + *(mask + m * 9 + (k + 1)*Mask_width + (l + 1))*
									(int)data_in[((i + k) * width + (j + l)) + layer*size];
#endif
							}
						}
					}
					else {
						for (int k = -1; k < 2; k++)
							for (int l = -1; l < 2; l++) {
								if (i + k >= 0 && i + k < height && j + l >= 0 && j + l < width)
#ifdef COLOR_INTERLEAVE
									sum =
									sum + *(mask + m * 9 + (k + 1)*channels + (l + 1))*
									(int)data_in[((i + k) * width + (j + l))*channels + layer];
#else
									sum =
									sum + *(mask + m * 9 + (k + 1)*Mask_width + (l + 1))*
									(int)data_in[((i + k) * width + (j + l)) + layer*size];
#endif
							}
					}
					max_sum = sum > max_sum ? sum : max_sum;
				}
#ifdef COLOR_INTERLEAVE
				data_out[(i * width + j)*channels + layer] = mymin(mymax(max_sum / 8, 0), 255);
#else
				data_out[(i * width + j) + layer*size] = mymin(mymax(max_sum / 8, 0), 255);
#endif
			}
		}
	}
}
