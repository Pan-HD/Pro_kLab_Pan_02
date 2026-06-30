// SegmentationConfig.h

#pragma once

//==================================================
// Segmentation Polarity
//
// 1:
//     Foreground = White (255)
//     Background = Black (0)
//
// 0:
//     Foreground = Black (0)
//     Background = White (255)
//==================================================

#define FOREGROUND_WHITE 1

#if FOREGROUND_WHITE

constexpr uchar FG_PIXEL = 255;
constexpr uchar BG_PIXEL = 0;

#else

constexpr uchar FG_PIXEL = 0;
constexpr uchar BG_PIXEL = 255;

#endif