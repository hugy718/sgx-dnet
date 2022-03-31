/*
 * Created on Wed Feb 19 2020
 *
 * Copyright (c) 2020 xxx xxx, xxxx
 */

#ifndef DATA_MNIST_H
#define DATA_MNIST_H

//use C linkage for these
extern "C"
{
#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"
#include "matrix.h"
#include <stdlib.h>
}

#include <cstring>
#include <cstdint>
#include <cstdio>

// basic file operations
#include <iostream>
#include <fstream>

#define ERROR()                                                                    \
    {                                                                              \
        std::cout << "I/O error in:" << __func__ << ": " << __LINE__ << std::endl; \
    }

uint32_t swap_bytes(uint32_t *);
data load_mnist_images(std::string path);
matrix load_mnist_labels(std::string path);
data load_one_mnist_image(std::string path);
matrix load_one_mnist_label();

#endif /* DATA_MNIST_H */
