#include "Util.h"
#include "ImageRenderer2.h"


ImageRenderer2::ImageRenderer2(void)
{
}


ImageRenderer2::~ImageRenderer2(void)
{
}


Mat ImageRenderer2::getImage()
{
	const double F		= this->lightfield.getRawFocalLength();	// focal length of the raw image
	const double beta	= focalLength / F;

	const int imageType = CV_MAKETYPE(CV_32F, this->lightfield.getRawImage().channels());
	Mat image(this->lightfield.SPARTIAL_RESOLUTION, imageType);
	Vec2f pinholePosition = Vec2f(this->pinholePosition);
	Vec2f spartialCorrection = Vec2f(this->lightfield.SPARTIAL_RESOLUTION.width, this->lightfield.ANGULAR_RESOLUTION.height) * -0.5;
	Vec2f angularCorrection = Vec2f(this->lightfield.ANGULAR_RESOLUTION.width, this->lightfield.ANGULAR_RESOLUTION.height) * -0.5;
	Vec2f pixelPosition, angularCoordinates;

	for (int y = 0; y < this->lightfield.SPARTIAL_RESOLUTION.height; y++)
	{
		for (int x = 0; x < this->lightfield.SPARTIAL_RESOLUTION.width; x++)
		{
			pixelPosition = Vec2f(x, y) + spartialCorrection;
			angularCoordinates = ((pinholePosition - pixelPosition) / beta) + pixelPosition;
			angularCoordinates -= angularCorrection;
			image.at<Vec3f>(Point(x, y)) = this->lightfield.getLuminance(x, y,
				round(angularCoordinates[0]), round(angularCoordinates[1]));
		}
	}

	return image;
}