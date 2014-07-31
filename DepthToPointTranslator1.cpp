#define _USE_MATH_DEFINES	// for math constants in C++
#include <opencv2\calib3d\calib3d.hpp>
#include "CDCDepthEstimator.h"
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
	Mat depthMap, cameraMatrix, roi, points;
	depth.download(depthMap);

	// 1) generate camera matrix
	// rotation and translation are 0 => the camera matrix P is reduced to the
	// calibration matrix K
	cameraMatrix = Mat(4, 4, CV_32FC1, Scalar(0));
	roi = cameraMatrix(Rect(0, 0, 3, 3));
	lightfield.getCalibrationMatrix().copyTo(roi);

	/*
	const float width = depth.size().width;
	const float height = depth.size().height;
	const float cx = 0;		// (cx, cy) is the optical center
	const float cy = 0;

	const float f = lightfield.loader.focalLength / lightfield.loader.pixelPitch;
	const float aspectRatio = height / width;
	const float fx = f;
	const float fy = aspectRatio * f;

	float K[4][4] = {
		{ fx,	0,	cx,	0 },
		{ 0,	fy,	cy,	0 },
		{ 0,	0,	1,	0 },
		{ 0,	0,	0,	1 }};
	cameraMatrix = Mat(4, 4, CV_32FC1, K);
	*/

	// 2) reproject image points with depth to 3d space
	reprojectImageTo3D(depthMap, points, cameraMatrix.inv());

	// 3) scale to proper size
	const float sx = 1.; // width / 2. * lightfield.loader.pixelPitch;
	const float sy = -1.; // height / 2. * lightfield.loader.pixelPitch * cos(M_PI / 3.;)
	const float sz = 1.;	// / (float) CDCDepthEstimator::ALPHA_MAX;
	const Scalar scale = Scalar(sx, sy, sz);
	points = points.mul(scale);

	return oclMat(points);
}