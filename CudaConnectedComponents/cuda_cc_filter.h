#pragma once

#include <cuda_runtime.h>
#include "cuda_cc_stats.h"

void computeRemoveMaskCUDA(
    const RegionStat* dStats,
    int maxLabel,
    int minArea,
    int maxArea,
    float minAspect,
    float maxAspect,
    unsigned char** dRemoveMaskOut
);

std::vector<unsigned char> downloadRemoveMaskCUDA(
    unsigned char* dRemoveMask,
    int maxLabel
);

void freeRemoveMaskCUDA(unsigned char* dRemoveMask);