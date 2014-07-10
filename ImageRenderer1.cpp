#define _USE_MATH_DEFINES	// for math constants in C++

#include "Util.h"
#include "ImageRenderer1.h"


const Vec2f ImageRenderer1::UV_SCALE = Vec2f(1.0, 1.0 / cos(M_PI / 6.0));
const float ImageRenderer1::ACCUMULATOR_SCALE = 1.2;


ImageRenderer1::ImageRenderer1(void)
{
}


ImageRenderer1::~ImageRenderer1(void)
{
}


void ImageRenderer1::setLightfield(LightFieldPicture lightfield)
{
	this->lightfield = lightfield;

	this->saSize = lightfield.SPARTIAL_RESOLUTION;
	this->imageSize = Size(saSize.width * ACCUMULATOR_SCALE,
		saSize.height * ACCUMULATOR_SCALE);
	this->imageType = CV_32FC2;//CV_MAKETYPE(CV_32F, this->lightfield.getRawImage().channels() + 1);
	this->angularCorrection = Vec2f(lightfield.ANGULAR_RESOLUTION.width, 
		lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	this->dstCenter = Vec2f(imageSize.width, imageSize.height) * 0.5;
	this->fromCenterToCorner = Vec2f(saSize.width, saSize.height) * -0.5;
	this->srcCorner	= dstCenter + fromCenterToCorner;
	this->srcCornerPoint = Point(round(srcCorner[0]), round(srcCorner[1]));
}


void ImageRenderer1::setAlpha(float alpha)
{
	this->alpha = alpha;
	this->weight = 1.0 - 1.0 / alpha;
}


oclMat ImageRenderer1::renderImage() const
{
	oclMat image = oclMat(imageSize, imageType, Scalar(0, 0));
	oclMat subapertureImage, dstROI;
	Vec2f angularIndices, realAngles, realTranslation, integralTranslation,
		srcAngles, dstTranslation, dstCorner;
	Rect dstRect;

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
			dstTranslation = round(realTranslation.mul(UV_SCALE));

			subapertureImage = lightfield.getSubapertureImageF(srcAngles[0], srcAngles[1]);

			appendRayCountingChannel(subapertureImage);

			dstCorner	= dstCenter + dstTranslation + fromCenterToCorner;
			dstRect		= Rect(Point(dstCorner), saSize);
			dstROI		= oclMat(image, dstRect);

			ocl::add(subapertureImage, dstROI, dstROI);
		}
	}

	// cut image to spartial resolution
	Rect srcRect	= Rect(srcCornerPoint, saSize);
	oclMat srcROI	= oclMat(image, srcRect);

	// scale luminance/color values to fit inside [0, 1]
	normalizeByRayCount(srcROI);

	return srcROI;
}