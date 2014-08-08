#define _USE_MATH_DEFINES	// for math constants in C++

#include "Util.h"
#include "ImageRenderer4.h"


ImageRenderer4::ImageRenderer4(void)
{
}


ImageRenderer4::~ImageRenderer4(void)
{
}


void ImageRenderer4::setLightfield(const LightFieldPicture& lightfield)
{
	this->lightfield = lightfield;
}


void ImageRenderer4::setAlpha(float alpha)
{
	this->weight = 1.0 - 1.0 / alpha;
}


oclMat ImageRenderer4::renderImage() const
{
	oclMat image = oclMat(lightfield.SPARTIAL_RESOLUTION, lightfield.IMAGE_TYPE,
		Scalar::all(0));
	oclMat rayCountAccumulator = oclMat(lightfield.SPARTIAL_RESOLUTION, CV_32FC1,
		Scalar::all(0));
	oclMat subapertureImage, modifiedSubapertureImage, rayCountMat;
	Mat transformation = Mat::eye(2, 3, CV_32FC1);

	if (weight > 1)
	{
		// TODO use interpolated sub-aperture images
	}
	else
	{
		int u, v;
		for(u = 0; u < this->lightfield.ANGULAR_RESOLUTION.width; u++)
		{
			transformation.at<float>(0, 2) = -(u - 5) * weight;
	
			for(v = 0; v < this->lightfield.ANGULAR_RESOLUTION.height; v++)
			{
				subapertureImage = lightfield.getSubapertureImageI(u, v);
				//normalize(subapertureImage);
	
				// shift sub-aperture image by (u, v) * (1 - 1 / alpha)
				transformation.at<float>(1, 2) = -(v - 5) * weight;
	
				ocl::warpAffine(subapertureImage, modifiedSubapertureImage,
					transformation, lightfield.SPARTIAL_RESOLUTION, INTER_CUBIC);

				rayCountMat = extractRayCountMat(modifiedSubapertureImage);
			
				ocl::add(modifiedSubapertureImage, image, image);
				ocl::add(rayCountMat, rayCountAccumulator, rayCountAccumulator);
			}
		}
	}

	// normalize each pixel by ray count
	normalizeByRayCount(image, rayCountAccumulator);
	normalize(image);

	return image;
}