#define _USE_MATH_DEFINES	// for math constants in C++

#include <iostream>	// debug
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Util.h"
#include "NormalDistribution.h"
#include "ImageRenderer3.h"


ImageRenderer3::ImageRenderer3(void)
{
}


ImageRenderer3::~ImageRenderer3(void)
{
}


oclMat ImageRenderer3::renderImage() const
{
	int x0 = this->getPinholePosition()[0]; // TODO kann außerhalb des ML-Bilds liegen
	int y0 = this->getPinholePosition()[1];

	float standardDeviation1 = this->lightfield.ANGULAR_RESOLUTION.width / 4.0;
	float standardDeviation2 = this->lightfield.ANGULAR_RESOLUTION.height / 4.0;
	NormalDistribution apertureFunction = NormalDistribution(x0, y0,
		standardDeviation1, standardDeviation2);

	const Vec2f uvScale = Vec2f(1.0, 1.0 / cos(M_PI / 6.0));

	const double weight = 1.0 - 1.0 / alpha;
	const Size saSize = Size(this->lightfield.SPARTIAL_RESOLUTION.width,
		this->lightfield.SPARTIAL_RESOLUTION.height);
	const Size imageSize = Size(saSize.width + this->lightfield.ANGULAR_RESOLUTION.width * uvScale[0] * weight,
		saSize.height + this->lightfield.ANGULAR_RESOLUTION.height * uvScale[1] * weight);
	const int imageType = CV_MAKETYPE(CV_32F, this->lightfield.getRawImage().channels()/* + 1*/);
	Mat image = Mat::zeros(imageSize, imageType);

	Mat subapertureImage, compositeImage, dstROI;
	Vec2d translation, dstCorner;
	const Vec2d angularCorrection = Vec2d(this->lightfield.ANGULAR_RESOLUTION.width,
		this->lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	const Vec2d dstCenter = Vec2d(image.size().width, image.size().height) * 0.5;
	Rect dstRect;
	const Vec2d fromCenterToCorner = Vec2d(this->lightfield.SPARTIAL_RESOLUTION.width,
		this->lightfield.SPARTIAL_RESOLUTION.height) * -0.5;

	int u, v;
	for(u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
	{
		for(v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
		{
			subapertureImage = this->lightfield.getSubapertureImageI(u, v);	// TODO reelle Koodinaten verwenden
			
			subapertureImage *= apertureFunction.f(u * uvScale[0] - angularCorrection[0],
				v * uvScale[1] - angularCorrection[1]);

			//compositeImage = appendRayCountingChannel(subapertureImage);
			
			translation	= (Vec2d(u * uvScale[0], v * uvScale[1]) - angularCorrection) * weight;
			dstCorner	= dstCenter + translation + fromCenterToCorner;
			dstRect		= Rect(Point(round(dstCorner[0]), round(dstCorner[1])), saSize);
			dstROI		= Mat(image, dstRect);

			add(subapertureImage, dstROI, dstROI, noArray(), imageType);
		}
	}

	// scale luminance/color values to fit inside [0, 1]
	//Mat normalizedImage = normalizeByRayCount(image);
	adjustLuminanceSpace(image);
	// TODO warum nicht CV::normalize()? Wenn ersetzbar, normalizeByRayCount() überflüssig

	return oclMat(image);
}