/*****************************************************************
/*Start Header
/*!
\file bmp.cpp
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

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "bmp.h"
//#define COLOR_INTERLEAVE 
#ifdef __linux
#define memcpy_s(dst, sizeofdst, src, count) memcpy(dst, src, count)
#endif

/* ************************************************************************* *
 *                                                                           *
 *  Opens a 24 bit true color bmp file and strips its header and its data.   *
 *  The data starts at location "data", its grouped into 3 layers of "size"  *
 *  bytes of size and they represent the colors blue, green and red.         *
 *                                                                           *
 * ************************************************************************* */

void bmp_read(char *filename, bmp_header * header, unsigned char **data)
{
    std::ifstream ifs(filename, std::ifstream::in | std::ios::binary);

    if (!ifs.is_open()) {
        std::
            cerr << "File " << filename << "cannot be opened." <<
            std::endl;
        exit(-1);
    };
    ifs.seekg(0, std::ios::end);
    unsigned int length_of_file = (unsigned int) ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    char *buffer = new char[length_of_file];
    ifs.read(buffer, length_of_file);
    memcpy_s(header, sizeof(bmp_header), buffer, sizeof(bmp_header));

    if (header->bits_per_pixel != 24) {
        std::cerr <<
            "Sorry, but can handle only 24-bit true color mode pictures."
            << std::endl;
        exit(-1);
    }

    unsigned int size = (*header).width * (*header).height;
    *data = new unsigned char[(3 * size * sizeof(unsigned char))];
    if ((*data) == NULL) {
        std::cerr << "Not enough memory for reading file!\n";
        exit(0);
    }

    char *ptr = buffer + sizeof(bmp_header);

    for (unsigned i = 0; i < size; i++) {
#ifdef COLOR_INTERLEAVE
		(*data)[i] = *ptr++;
		(*data)[i + 1] = *ptr++;
		(*data)[i + 2] = *ptr++;
#else
		(*data)[i] = *ptr++;
		(*data)[i + size] = *ptr++;
		(*data)[i + 2 * size] = *ptr++;
#endif
		if ((i + 1) % header->width == 0) {
			int j = 1;
			while ((i + j) % 4) {
				j++;
				*ptr++;
			}	
		//	ptr += i % 4;
		}
    }

    delete[]buffer;

    ifs.close();
};



/* ************************************************************************* *
 *                                                                           *
 *  Stores a 24 bit true color bmp file given in the format described above. *
 *                                                                           *
 * ************************************************************************* */

extern void
bmp_write(char *filename, bmp_header * header, unsigned char *data)
{
    std::ofstream ofs(filename, std::ofstream::out | std::ios::binary);
    unsigned long int size, i, j;

    unsigned row_size =
        ((header->bits_per_pixel * header->width + 31) / 32) * 4;
    unsigned pixelarraysize = row_size * header->height;

    size = header->width * header->height;

    if (!ofs.is_open()) {
        std::cerr << "File " << filename << " couldn't be opened\n" <<
            std::endl;
        exit(-1);
    }

    char *buffer = new char[pixelarraysize + sizeof(bmp_header)];

    memcpy_s(buffer, size + sizeof(bmp_header), header,
             sizeof(bmp_header));

    char *ptr = buffer + sizeof(bmp_header);
    for (i = 0; i < size; i ++) {
#ifdef COLOR_INTERLEAVE
		*ptr++ = data[i];
		*ptr++ = data[i + 1];
		*ptr++ = data[i + 2];
#else
		*ptr++ = data[i];
		*ptr++ = data[i + size];
		*ptr++ = data[i + 2 * size];
#endif
		if ((i + 1) % header->width == 0) {
			j = 1;
			while ((i + j) % 4) {
				j++;
				*ptr++ = 0;
			}
		}
    }
    ofs.write(buffer, pixelarraysize + sizeof(bmp_header));
    ofs.close();
    delete[]data;
};
