#define _USE_MATH_DEFINES	// for math constants in C++
#include <string>
#include <cmath>
#include <iostream>	// debug
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ocl/ocl.hpp>
#include "Util.h"
#include "LfpLoader.h"
#include "LightFieldPicture.h"


const int LightFieldPicture::IMAGE_TYPE = CV_32FC1;
const LightFieldPicture::luminanceType LightFieldPicture::ZERO_LUMINANCE = 0;

const Point LightFieldPicture::IMAGE_ORIGIN = Point(0, 0);


void LightFieldPicture::extractSubapertureImageAtlas()
{
	const double pixelPitch		= loader.pixelPitch;
	const double scaleFactorX	= loader.scaleFactor[0];
	const double scaleFactorY	= loader.scaleFactor[1];

	const int dstWidth	= ANGULAR_RESOLUTION.width * SPARTIAL_RESOLUTION.width;
	const int dstHeight	= ANGULAR_RESOLUTION.height * SPARTIAL_RESOLUTION.height;

	const int u0 = ANGULAR_RESOLUTION.width / 2;
	const int v0 = ANGULAR_RESOLUTION.height / 2;
	const int s0 = SPARTIAL_RESOLUTION.width / 2;
	const int t0 = SPARTIAL_RESOLUTION.height / 2 - 1;

	typedef Vec2f coord;
	Mat_<coord> map = Mat(dstHeight, dstWidth, CV_32FC2);
	int x, y, s, t, u, v;
	coord tmp;
	for (y = 0; y < dstHeight; y++)
	{
		v = y / SPARTIAL_RESOLUTION.height - v0;
		t = y % SPARTIAL_RESOLUTION.height - t0;

		tmp = mlaCenter + t * nextRow;

		for (x = 0; x < dstWidth; x++)
		{
			u = x / SPARTIAL_RESOLUTION.width - u0;
			s = x % SPARTIAL_RESOLUTION.width - s0;

			map.at<coord>(y, x) = tmp + floor(s - t / 2.) * nextLens + Vec2f(u, v);
		}
	}

	ocl::remap(oclMat(this->rawImage), this->subapertureImageAtlas, oclMat(map), oclMat(),
		CV_INTER_LINEAR, BORDER_CONSTANT, this->ZERO_LUMINANCE);
}


LightFieldPicture::LightFieldPicture(void)
{
}


LightFieldPicture::LightFieldPicture(const std::string& pathToFile)
{
	// load raw data
	this->loader	= LfpLoader(pathToFile);

	this->lensPitchInPixels			= loader.lensPitch / loader.pixelPitch;
	this->rotationAngle				= loader.rotationAngle;
	this->microLensRadiusInPixels	= lensPitchInPixels / 2.;

	// calculate angular resolution of the light field
	const int lensPitch			= ceil(lensPitchInPixels);
	this->ANGULAR_RESOLUTION	= Size(lensPitch, lensPitch);

	// calculate spartial resolution of the light field
	const Size correctedRawSize = RotatedRect(IMAGE_ORIGIN, loader.bayerImage.size(),
		-this->rotationAngle).boundingRect().size();
	const double lensWidth		= lensPitchInPixels * loader.scaleFactor[0];
	const double rowHeight		= lensPitchInPixels * loader.scaleFactor[1] * cos(M_PI / 6.0);
	const int columnCount		= ceil(correctedRawSize.width / lensWidth) + 1;
	const int rowCount			= ceil(correctedRawSize.height / rowHeight) + 1;
	this->SPARTIAL_RESOLUTION	= Size(columnCount, rowCount);

	this->validSpartialCoordinates	= Rect(IMAGE_ORIGIN, SPARTIAL_RESOLUTION);
	this->fromLensCenterToOrigin	= Vec2f(ANGULAR_RESOLUTION.width,
		ANGULAR_RESOLUTION.height) * -0.5;

	const Size rawSize			= loader.bayerImage.size();
	const Vec2f sensorCenter	= Vec2f(rawSize.width, rawSize.height) * 0.5;
	const Vec2f mlaOffset		= Vec2f(loader.sensorOffset[0],
		loader.sensorOffset[1]) / loader.pixelPitch;
	this->mlaCenter				= sensorCenter + mlaOffset;

	const double angleToNextRow = M_PI / 3.0 + rotationAngle;	// 60° to MLA axis
	this->nextLens = Vec2f(cos(rotationAngle), sin(rotationAngle))
		* lensPitchInPixels;
	this->nextRow = Vec2f(cos(angleToNextRow), sin(angleToNextRow))
		* lensPitchInPixels;

	// process raw image
	Mat demosaicedImage;
	//cvtColor(loader.bayerImage, demosaicedImage, CV_BayerBG2RGB);
	cvtColor(loader.bayerImage, demosaicedImage, CV_BayerBG2GRAY);
	demosaicedImage.convertTo(demosaicedImage, IMAGE_TYPE, 1.0 / 65535.0);
	
	demosaicedImage.copyTo(this->rawImage);

	// get subaperture images
	extractSubapertureImageAtlas();

	size_t saImageCount = ANGULAR_RESOLUTION.area();
	subapertureImages = vector<oclMat>(saImageCount);

	Point imageCorner;
	Rect imageRect;
	oclMat subapertureImage;

	int u, v, index = 0;
	for (v = 0; v < ANGULAR_RESOLUTION.height; v++)
	{
		for (u = 0; u < ANGULAR_RESOLUTION.width; u++)
		{
			imageCorner = Point(u * this->SPARTIAL_RESOLUTION.width,
				v * this->SPARTIAL_RESOLUTION.height);
			imageRect = Rect(imageCorner, this->SPARTIAL_RESOLUTION);
			subapertureImage = oclMat(this->subapertureImageAtlas, imageRect);

			subapertureImages[index] = subapertureImage;
			index++;
		}
	}
}


LightFieldPicture::~LightFieldPicture(void)
{
	this->rawImage.release();
}


LightFieldPicture::luminanceType LightFieldPicture::getLuminanceI(
	const int x, const int y, const int u, const int v) const
{
	// luminance outside of the recorded spartial range is zero
	if (!validSpartialCoordinates.contains(Point(x, y)))
		return ZERO_LUMINANCE;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	Vec2f uv = Vec2d(u, v);
	uv += fromLensCenterToOrigin;	// center value range at (0, 0)
	double nrm = norm(uv);
	if (nrm > microLensRadiusInPixels)
	{
		uv *= microLensRadiusInPixels / nrm;
		uv = roundToZero(uv);
	}
	uv -= fromLensCenterToOrigin;

	Vec2f pixelPosition = mlaCenter + y * nextRow + (x - y / 2) * nextLens + uv;
	return this->rawImage.at<luminanceType>(Point(pixelPosition));
}


LightFieldPicture::luminanceType LightFieldPicture::getLuminanceF(
	const float x, const float y, const float u, const float v) const
{
	// handle coordinates outside of the recorded lightfield
	const float halfWidth = SPARTIAL_RESOLUTION.width / 2.0;
	const float halfHeight = SPARTIAL_RESOLUTION.height / 2.0;
	// luminance outside the recorded spartial range is zero
	if (abs(x) > halfWidth || abs(y) > halfHeight)
		return ZERO_LUMINANCE;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	Vec2f angularVector = Vec2f(u, v);
	double nrm = norm(angularVector);
	if (nrm > microLensRadiusInPixels)
	{
		//angularVector *= microLensRadiusInPixels / nrm;
		normalize(angularVector, angularVector, microLensRadiusInPixels);
		return getLuminanceI(x, y, angularVector[0], angularVector[1]);
	}

	/*
	const unsigned int rawX = x * this->ANGULAR_RESOLUTION.width + u;
	const unsigned int rawY = y * this->ANGULAR_RESOLUTION.height + v;

	return this->rawImage.at<luminanceType>(Point(rawX, rawY));
	*/

	Mat singlePixel;
	const Size singlePixelSize = Size(1, 1);
	Vec2f lensSize = Vec2f(this->ANGULAR_RESOLUTION.width, this->ANGULAR_RESOLUTION.height);
	Vec2f centralLensCenter = Vec2f(this->SPARTIAL_RESOLUTION.width,
		this->SPARTIAL_RESOLUTION.height) / 2.0; // muss eigentlich auf ein Vielfaches der Linsengröße gerundet werden
	Vec2f lensCenter = centralLensCenter + Vec2f(x, y).mul(lensSize);
	lensCenter = Vec2f(round(lensCenter[0]), round(lensCenter[1]));
	Vec2f position = lensCenter + angularVector;
	getRectSubPix(rawImage, singlePixelSize, Point2f(position), singlePixel);

	return singlePixel.at<luminanceType>(Point(0, 0));
}


oclMat LightFieldPicture::getSubapertureImageI(const unsigned short u,
	const unsigned short v) const
{
	return this->subapertureImages[v * this->ANGULAR_RESOLUTION.width + u];
}


oclMat LightFieldPicture::getSubapertureImageF(const double u, const double v) const
{
	// TODO Koordinaten außerhalb des Linsenbildes besser behandeln
	const int minAngle = 0;
	const int maxAngle = ANGULAR_RESOLUTION.width - 1;
	int fu = min(maxAngle, max(minAngle, (int) floor(u)));
	int cu = min(maxAngle, max(minAngle, (int) ceil(u)));
	int fv = min(maxAngle, max(minAngle, (int) floor(v)));
	int cv = min(maxAngle, max(minAngle, (int) ceil(v)));

	oclMat upperLeftImage	= this->subapertureImages[
		fv * this->ANGULAR_RESOLUTION.width + fu];
	oclMat lowerLeftImage	= this->subapertureImages[
		cv * this->ANGULAR_RESOLUTION.width + fu];
	oclMat upperRightImage	= this->subapertureImages[
		fv * this->ANGULAR_RESOLUTION.width + cu];
	oclMat lowerRightImage	= this->subapertureImages[
		cv * this->ANGULAR_RESOLUTION.width + cu];

	float lowerWeight	= v - floor(v);
	float upperWeight	= 1.0 - lowerWeight;
	float rightWeight	= u - floor(u);
	float leftWeight	= 1.0 - rightWeight;

	float upperLeftWeight	= upperWeight * leftWeight;
	float lowerLeftWeight	= lowerWeight * leftWeight;
	float upperRightWeight	= upperWeight * rightWeight;
	float lowerRightWeight	= lowerWeight * rightWeight;

	oclMat leftSum, rightSum, totalSum;
	ocl::addWeighted(upperLeftImage, upperLeftWeight,
		lowerLeftImage, lowerLeftWeight, 0, leftSum);
	ocl::addWeighted(upperRightImage, upperRightWeight,
		lowerRightImage, lowerRightWeight, 0, rightSum);
	ocl::add(leftSum, rightSum, totalSum);

	return totalSum;
}


Mat LightFieldPicture::getRawImage() const
{
	return this->rawImage;
}


oclMat LightFieldPicture::getSubapertureImageAtlas() const
{
	return this->subapertureImageAtlas;
}


double LightFieldPicture::getRawFocalLength() const
{
	return this->loader.focalLength;
}
