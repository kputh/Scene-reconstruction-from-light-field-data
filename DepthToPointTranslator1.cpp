#define _USE_MATH_DEFINES	// for math constants in C++
#include <opencv2\calib3d\calib3d.hpp>
#include "DepthToPointTranslator1.h"


DepthToPointTranslator1::DepthToPointTranslator1(void)
{
}


DepthToPointTranslator1::~DepthToPointTranslator1(void)
{
}


oclMat DepthToPointTranslator1::translateDepthToPoints(const oclMat& depth,
	const LightFieldPicture& lightfield) const
{
	Mat depthMap, cameraMatrix, points;
	depth.download(depthMap);

	// 1) generate camera matrix
	// rotation and translation are 0 => the camera matrix P is reduced to the
	// calibration matrix K
	const float width = depth.size().width;
	const float height = depth.size().height;
	const float cx = width / 2.;	// (cx, cy) is the optical center)
	const float cy = height / 2.;

	const float fx = lightfield.getRawFocalLength();	// fx = f
	const float fy = height / width * fx;				// fy = aspectRatio * f

	float K[4][4] = {
		{ fx,	0,	cx,	0 },
		{ 0,	fy,	cy,	0 },
		{ 0,	0,	1,	0 },
		{ 0,	0,	0,	1 }};
	cameraMatrix = Mat(4, 4, CV_32FC1, K);

	// 2) reproject image points with depth to 3d space
	reprojectImageTo3D(depthMap, points, cameraMatrix.inv());

	// 3) scale to compensate the shape of the microlens array
	const float sx = lightfield.loader.scaleFactor[0];
	const float sy = lightfield.loader.scaleFactor[1] * cos(M_PI / 3.);
	float S[3][3] = {
		{ sx,	0,	0 },
		{ 0,	sy,	0 },
		{ 0,	0,	1 }};
	Mat scalingMatrix = Mat(3, 3, CV_32FC1, S);

	points = scalingMatrix * points;

	return oclMat(points);
}

Mat DepthToPointTranslator1::makeCalibrationMatrix(
	const LightFieldPicture& lightfield, const Size& imageSize) const
{
	/*
	const float focalLengthInPixels = lightfield.loader.focalLength /
		this->loader.pixelPitch;
	const float opticalCenterX = lightfield.loader.bayerImage.size().width / 2.;
	const float opticalCenterX = lightfield.loader.bayerImage.size().height / 2.;

	float K[3][3] = {
		{focalLengthInPixels,	0,						opticalCenterX},
		{0,						focalLengthInPixels,	opticalCenterY},
		{0,						0,						1}};
	Mat calibrationMatrix = Mat(3, 3, CV_32FC1, K);
	*/
	return Mat();
}
