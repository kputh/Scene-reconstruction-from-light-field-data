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
	const Size imageSize = Size(saSize.width + this->lightfield.ANGULAR_RESOLUTION.width * uvScale[0] * weight,
		saSize.height + this->lightfield.ANGULAR_RESOLUTION.height * uvScale[1] * weight);
	const int imageType = CV_MAKETYPE(CV_32F, this->lightfield.getRawImage().channels());
	Mat image = Mat::zeros(imageSize, imageType);

	Mat subapertureImage, compositeImage, dstROI;
	Vec2d translation, dstCorner;
	const Vec2d angularCorrection = Vec2d(this->lightfield.ANGULAR_RESOLUTION.width,
		this->lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	const Vec2d dstCenter = Vec2d(image.size().width, image.size().height) * 0.5;
	Rect dstRect;
	const Vec2d fromCenterToCorner = Vec2d(saSize.width, saSize.height) * -0.5;
	const int interpolationMethod = (alpha < 0.0) ? CV_INTER_AREA : CV_INTER_CUBIC;

	for(int u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
	{
		for(int v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
		{
			subapertureImage = this->lightfield.getSubapertureImage(u, v);

			compositeImage = appendRayCountingChannel(subapertureImage);

			translation	= (Vec2d(u * uvScale[0], v * uvScale[1]) - angularCorrection) * weight;
			dstCorner	= dstCenter + translation + fromCenterToCorner;
			dstRect		= Rect(Point(round(dstCorner[0]), round(dstCorner[1])), saSize);
			dstROI		= Mat(image, dstRect);

			add(compositeImage, dstROI, dstROI, noArray(), dstROI.type());
		}
	}

	// scale luminance/color values to fit inside [0, 1]
	Mat normalizedImage = normalizeByRayCount(image);

	return normalizedImage;
}
