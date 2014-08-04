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


Mat DepthToPointTranslator1::translateDepthToPoints(const Mat& depthMap,
	const Mat& calibrationMatrix, const Mat& rotation, const Mat& translation)
	const
{
	Mat K44, Rt44, cameraMatrix, roi, points;
	const Mat identityMatrix = Mat::eye(4, 4, CV_64FC1);
	const Mat zeroRow = identityMatrix.row(3);		// TODO constants
	const Mat zeroColumn = Mat::zeros(3, 1, CV_64FC1);

	// 1) generate 4x4 camera matrix
	// combine rotation and translation into a single 4x4 matrix
	hconcat(rotation, translation, Rt44);
	vconcat(Rt44, zeroRow, Rt44);

	// expand the calibration matrix into a 4x4 matrix
	hconcat(calibrationMatrix, zeroColumn, K44);
	vconcat(K44, zeroRow, K44);

	// P = K[R|t]
	cameraMatrix = K44 * Rt44;	

	// 2) reproject image points with depth to 3D space
	reprojectImageTo3D(depthMap, points, cameraMatrix.inv());

	// 3) scale to proper size
	const float sx = 1.; // width / 2. * lightfield.loader.pixelPitch;
	const float sy = -1.; // height / 2. * lightfield.loader.pixelPitch * cos(M_PI / 3.;)
	const float sz = 1.;	// / (float) CDCDepthEstimator::ALPHA_MAX;
	const Scalar scale = Scalar(sx, sy, sz);
	points = points.mul(scale);

	return points;
}