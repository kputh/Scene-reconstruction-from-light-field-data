#pragma once

#include <string>
#include <opencv2/core/core.hpp>

#include "LightFieldFromLfpFile.h"

double round(double value);
double roundTo(double value, double target);
double roundToZero(double value);
Mat adjustLuminanceSpace(const Mat image);
void saveImageToPNGFile(string fileName, Mat image);
void saveImageArc(LightFieldFromLfpFile lightfield, string sourceFileName, int imageCount);