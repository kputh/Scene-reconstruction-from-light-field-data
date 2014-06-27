#pragma once

#include <string>
#include <opencv2/core/core.hpp>

#include "LightFieldPicture.h"

double round(double value);
double roundTo(double value, double target);
double roundToZero(double value);
Mat adjustLuminanceSpace(const Mat image);
void saveImageToPNGFile(string fileName, Mat image);
void saveImageArc(LightFieldPicture lightfield, string sourceFileName, int imageCount);
Mat appendRayCountingChannel(Mat image);
Mat normalizeByRayCount(Mat image);