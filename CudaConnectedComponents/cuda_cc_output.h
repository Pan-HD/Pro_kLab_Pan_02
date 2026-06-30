#pragma once

#include <cuda_runtime.h>

void generateFilteredBinaryCUDA(
    const unsigned int* dLabel,
    const unsigned char* dRemoveMask,
    int width,
    int height,
    unsigned char** dOutputOut
);

void freeFilteredBinaryCUDA(unsigned char* dOutput);