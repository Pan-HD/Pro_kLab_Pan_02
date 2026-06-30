#pragma once

#include <cuda_runtime.h>
#include <vector>

struct RegionStat
{
    int area;
    int xmin;
    int xmax;
    int ymin;
    int ymax;
};

void computeRegionStatsCUDA(
    const unsigned int* dLabel,
    int width,
    int height,
    int maxLabel,
    RegionStat** dStatsOut
);

std::vector<RegionStat> downloadRegionStatsCUDA(
    RegionStat* dStats,
    int maxLabel
);

void freeRegionStatsCUDA(RegionStat* dStats);