#define _USE_MATH_DEFINES	// for math constants in C++

#include "Util.h"
#include "ImageRenderer1.h"


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

	Size saSize = lightfield.SPARTIAL_RESOLUTION;
	this->imageSize = Size(saSize.width * ACCUMULATOR_SCALE,
		saSize.height * ACCUMULATOR_SCALE);
	this->imageType = CV_MAKETYPE(CV_32F,
		this->lightfield.getSubapertureImageAtlas().channels() + 1);
	this->angularCorrection = Vec2f(lightfield.ANGULAR_RESOLUTION.width, 
		lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	this->fromCornerToCenter = Vec2f(imageSize.width - saSize.width,
		imageSize.height - saSize.height) * 0.5;

	const int cutWidth = saSize.width - 4;
	const int cutHeight = saSize.height - 4;
	this->cutRect = Rect((imageSize.width - cutWidth) / 2,
		(imageSize.height - cutHeight) / 2, cutWidth, cutHeight);
}


void ImageRenderer1::setAlpha(float alpha)
{
	this->alpha = alpha;
	this->weight = 1.0 - 1.0 / alpha;
}


oclMat ImageRenderer1::renderImage() const
{
	if (alpha == 1)
	{
		oclMat image = oclMat(lightfield.SPARTIAL_RESOLUTION,
			lightfield.IMAGE_TYPE, Scalar::all(0));

		oclMat subapertureImage;
		int u, v;
		for(u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
			for(v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
			{
				subapertureImage = lightfield.getSubapertureImageI(u, v);
				ocl::add(subapertureImage, image, image);
			}

		image /= lightfield.ANGULAR_RESOLUTION.area();
		normalize(image);
		return image;
	}

	oclMat image = oclMat(imageSize, lightfield.IMAGE_TYPE, Scalar::all(0));
	oclMat rayCountAccumulator = oclMat(imageSize, CV_32FC1, Scalar::all(0));
	oclMat subapertureImage, modifiedSubapertureImage, rayCountMat;
	Vec2f translation;
	Point2f dstTri[3];
	Mat transformation;

	int u, v;
	for(u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
	{
		for(v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
		{
			subapertureImage = lightfield.getSubapertureImageI(u, v);
			//normalize(subapertureImage);

			// shift sub-aperture image by (u, v) * (1 - 1 / alpha) from center
			translation = (Vec2f(u, v) - angularCorrection) * weight;
			translation += this->fromCornerToCenter;
			dstTri[0] = Point2f(0 + translation[0], 0 + translation[1]);
			dstTri[1] = Point2f(1 + translation[0], 0 + translation[1]);
			dstTri[2] = Point2f(0 + translation[0], 1 + translation[1]);
			transformation = getAffineTransform(UNIT_VECTORS, dstTri);

			ocl::warpAffine(subapertureImage, modifiedSubapertureImage,
				transformation, imageSize, INTER_LINEAR);

			rayCountMat = extractRayCountMat(modifiedSubapertureImage);
			
			ocl::add(modifiedSubapertureImage, image, image);
			ocl::add(rayCountMat, rayCountAccumulator, rayCountAccumulator);
		}
	}

	// normalize each pixel by ray count
	normalizeByRayCount(image, rayCountAccumulator);
	normalize(image);

	/*
	// cut image to spartial resolution
	oclMat srcROI	= oclMat(image, cutRect);
	oclMat cutImage;	srcROI.copyTo(cutImage);

	return cutImage;
	*/
	return image;
}