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
	// TODO
	return oclMat();
}

Mat DepthToPointTranslator1::makeCalibrationMatrix(
	const LightFieldPicture& lightfield, const Size& imageSize) const
{
	const float focalLengthInPixels = this->loader.focalLength /
		this->loader.pixelPitch;
	const float opticalCenterX = this->loader.bayerImage.size().width / 2.;
	const float opticalCenterX = this->loader.bayerImage.size().height / 2.;

	float[3][3] K = {
		{focalLengthInPixels,	0,						opticalCenterX},
		{0,						focalLengthInPixels,	opticalCenterY},
		{0,						0,						1}};
	Mat calibrationMatrix = Mat(3, 3, CV_32FC1, K);

}
