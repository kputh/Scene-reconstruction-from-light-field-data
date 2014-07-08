#define _USE_MATH_DEFINES	// for math constants in C++

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ocl/ocl.hpp>
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
	return Mat();
}


oclMat ImageRenderer1::renderImage()
{
	const Vec2f uvScale = Vec2f(1.0, 1.0 / cos(M_PI / 6.0));

	const double weight = 1.0 - 1.0 / alpha;
	const Size saSize = Size(this->lightfield.SPARTIAL_RESOLUTION.width,
		this->lightfield.SPARTIAL_RESOLUTION.height);
	const Size imageSize = Size(saSize.width + ceil(abs(this->lightfield.ANGULAR_RESOLUTION.width * uvScale[0] * weight)),
		saSize.height + ceil(abs(this->lightfield.ANGULAR_RESOLUTION.height * uvScale[1] * weight)));
	const int imageType = CV_MAKETYPE(CV_32F, this->lightfield.getRawImage().channels() + 1);
	oclMat image = oclMat(imageSize, imageType, Scalar(0, 0, 0));

	oclMat subapertureImage, dstROI;
	Vec2f angularIndices, realAngles, realTranslation, integralTranslation,
		srcAngles, dstTranslation, dstCorner;
	const Vec2f angularCorrection = Vec2f(this->lightfield.ANGULAR_RESOLUTION.width, 
		this->lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	const Vec2f dstCenter = Vec2f(image.size().width, image.size().height) * 0.5;
	Rect dstRect;
	const Vec2f fromCenterToCorner = Vec2f(saSize.width, saSize.height) * -0.5;

	int u, v;
	for(u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
	{
		for(v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
		{
			angularIndices = Vec2f(u, v);
			realAngles = angularIndices - angularCorrection;
			realTranslation = realAngles * weight;
			integralTranslation = round(realTranslation);
				// Vec2f(floor(realTranslation[0]), floor(realTranslation[1]));
			srcAngles = angularIndices -
				(realTranslation - integralTranslation) / weight;
			dstTranslation = round(realTranslation.mul(uvScale));

			subapertureImage = this->lightfield.getSubapertureImageF(srcAngles[0], srcAngles[1]);

			appendRayCountingChannel(subapertureImage);

			dstCorner	= dstCenter + dstTranslation + fromCenterToCorner;
			dstRect		= Rect(Point(dstCorner), saSize);
			dstROI		= oclMat(image, dstRect);

			ocl::add(subapertureImage, dstROI, dstROI);
		}
	}

	// cut image to spartial resolution
	Vec2f scrCorner	= dstCenter + fromCenterToCorner;
	Rect srcRect	= Rect(Point(round(scrCorner[0]), round(scrCorner[1])), saSize);
	oclMat srcROI	= oclMat(image, srcRect);

	// scale luminance/color values to fit inside [0, 1]
	normalizeByRayCount(srcROI);

	return srcROI;
}