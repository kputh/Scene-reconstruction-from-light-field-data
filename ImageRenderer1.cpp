#define _USE_MATH_DEFINES	// for math constants in C++

#include <opencv2/imgproc/imgproc.hpp>
#include "Util.h"
#include "ImageRenderer1.h"


ImageRenderer1::ImageRenderer1(void)
{
}


ImageRenderer1::~ImageRenderer1(void)
{
}


Mat ImageRenderer1::renderImage()
{
	const double F		= this->lightfield.getRawFocalLength();	// focal length of the raw image
	const double alpha	= focalLength / F;

	const Vec2f uvScale = Vec2f(1.0, 1.0 / cos(M_PI / 6.0));

	const double weight = 1.0 - 1.0 / alpha;
	const Size saSize = Size(this->lightfield.SPARTIAL_RESOLUTION.width,
		this->lightfield.SPARTIAL_RESOLUTION.height);
	const Size imageSize = Size(saSize.width + abs(ceil(this->lightfield.ANGULAR_RESOLUTION.width * uvScale[0] * weight)),
		saSize.height + abs(ceil(this->lightfield.ANGULAR_RESOLUTION.height * uvScale[1] * weight)));
	const int imageType = CV_MAKETYPE(CV_32F, this->lightfield.getRawImage().channels() + 1);
	Mat image = Mat::zeros(imageSize, imageType);

	Mat subapertureImage, compositeImage, dstROI;
	Vec2d translation, dstCorner;
	const Vec2d angularCorrection = Vec2d(this->lightfield.ANGULAR_RESOLUTION.width,
		this->lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	const Vec2d dstCenter = Vec2d(image.size().width, image.size().height) * 0.5;
	Rect dstRect;
	const Vec2d fromCenterToCorner = Vec2d(saSize.width, saSize.height) * -0.5;

	int u, v;
	for(u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
	{
		for(v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
		{
			subapertureImage = this->lightfield.getSubapertureImage(u, v);

			compositeImage = appendRayCountingChannel(subapertureImage);

			translation	= (Vec2d(u, v) - angularCorrection).mul(uvScale) * weight;
			dstCorner	= dstCenter + translation + fromCenterToCorner;
			dstRect		= Rect(Point(round(dstCorner[0]), round(dstCorner[1])), saSize);
			dstROI		= Mat(image, dstRect);

			add(compositeImage, dstROI, dstROI, noArray(), imageType);
		}
	}

	// cut image to spartial resolution
	Vec2f scrCorner	= dstCenter + fromCenterToCorner;
	Rect srcRect	= Rect(Point(round(scrCorner[0]), round(scrCorner[1])), saSize);
	Mat srcROI		= Mat(image, srcRect);

	// scale luminance/color values to fit inside [0, 1]
	Mat normalizedImage = normalizeByRayCount(srcROI);

	return normalizedImage;
}
