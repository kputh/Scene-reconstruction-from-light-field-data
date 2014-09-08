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
	//reprojectImageTo3D(depthMap, points, cameraMatrix.inv());

	const float lensPitch			= 1.389859962463379e-005;	// TODO get from LightFieldPicture
	const float horizontalDistance	= lensPitch;
	const float verticalDistance	= lensPitch * cos(M_PI / 6.);

	points = Mat(depthMap.size(), CV_32FC3);
	for (int y = 0; y < depthMap.rows; y++)
	for (int x = 0; x < depthMap.cols; x++)
		points.at<Vec3f>(y, x) = Vec3f(
		x * horizontalDistance, y * verticalDistance, depthMap.at<float>(y, x));

	perspectiveTransform(points, points, cameraMatrix.inv());

	// 3) flip around x axis
	const Scalar scale = Scalar(1, -1, 1);
	points = points.mul(scale);

	return points;
}