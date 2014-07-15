#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/ocl/ocl.hpp>

#include "LightFieldPicture.h"

double round(double value);
Vec2d round(Vec2d vector);
double roundTo(double value, double target);
double roundToZero(double value);
Vec2d roundToZero(Vec2d vector);
void adjustLuminanceSpace(Mat& image);
void saveImageToPNGFile(string fileName, Mat image);
void saveImageArc(LightFieldPicture lightfield, string sourceFileName,
	int imageCount);
void appendRayCountingChannel(Mat& image);
void normalizeByRayCount(Mat& image);
void appendRayCountingChannel(oclMat& image);
void normalizeByRayCount(oclMat& image);
oclMat extractRayCountMat(const oclMat& image);
void normalizeByRayCount(oclMat& image, const oclMat& rayCountMat);
