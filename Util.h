#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/ocl/ocl.hpp>

#include "LightFieldPicture.h"
#include "CameraPoseEstimator.h"

const Point2f UNIT_VECTORS[3] = { Point2f(0, 0), Point2f(1, 0),
	Point2f(0, 1) };

double round(double value);
Vec2d round(Vec2d vector);
double roundTo(double value, double target);
double roundToZero(double value);
Vec2d roundToZero(Vec2d vector);
void adjustLuminanceSpace(Mat& image);
void saveImageToPNGFile(string fileName, Mat image);
void appendRayCountingChannel(Mat& image);
void normalizeByRayCount(Mat& image);
void appendRayCountingChannel(oclMat& image);
void normalizeByRayCount(oclMat& image);
oclMat extractRayCountMat(const oclMat& image);
void normalizeByRayCount(oclMat& image, const oclMat& rayCountMat);
void normalize(oclMat& mat);

// debugging functions
void saveImageArc(LightFieldPicture lightfield, string sourceFileName,
	int imageCount);
void visualizeCameraTrajectory(const CameraPoseEstimator& estimator,
	const Matx33d& calibrationMatrix);