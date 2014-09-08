#include "Util.h"
#include "ImageRenderer2.h"


ImageRenderer2::ImageRenderer2(void)
{
}


ImageRenderer2::~ImageRenderer2(void)
{
}


oclMat ImageRenderer2::renderImage() const
{
	float beta = alpha;

	const int imageType = CV_MAKETYPE(CV_32F,
		this->lightfield.getRawImage().channels());
	Mat image(this->lightfield.SPARTIAL_RESOLUTION, imageType);
	Vec2f pinholePosition = Vec2f(this->pinholePosition);
	Vec2f spartialCorrection = Vec2f(this->lightfield.SPARTIAL_RESOLUTION.width,
		this->lightfield.ANGULAR_RESOLUTION.height) * -0.5;
	Vec2f angularCorrection = Vec2f(this->lightfield.ANGULAR_RESOLUTION.width,
		this->lightfield.ANGULAR_RESOLUTION.height) * -0.5;
	Vec2f pixelPosition, angularCoordinates;

	int x, y;
	for (y = 0; y < this->lightfield.SPARTIAL_RESOLUTION.height; y++)
	{
		for (x = 0; x < this->lightfield.SPARTIAL_RESOLUTION.width; x++)
		{
			pixelPosition = Vec2f(x, y) + spartialCorrection;
			angularCoordinates = ((pinholePosition - pixelPosition) / beta) +
				pixelPosition;
			angularCoordinates -= angularCorrection;
			image.at<Vec3f>(Point(x, y)) = this->lightfield.getLuminanceI(x, y,
				round(angularCoordinates[0]), round(angularCoordinates[1]));
		}
	}

	return oclMat(image);
}