#define _USE_MATH_DEFINES	// for math constants in C++
#include <string>
#include <cmath>
#include <iostream>	// debug
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Util.h"
#include "LfpLoader.h"
#include "LightFieldFromLfpFile.h"


const int LightFieldFromLfpFile::IMAGE_TYPE = CV_32FC3;


Mat LightFieldFromLfpFile::convertBayer2RGB(const Mat bayerImage)
{
	Mat colorImage(bayerImage.size(), CV_16UC3);
	cvtColor(bayerImage, colorImage, CV_BayerBG2RGB);
	return colorImage;
}


Mat LightFieldFromLfpFile::rectifyLensGrid(const Mat hexagonalLensGrid, LfpLoader metadata)
{
	Mat inputImage;
	hexagonalLensGrid.convertTo(inputImage, LightFieldFromLfpFile::IMAGE_TYPE);

	// 1) read LFP metadata
	const double pixelPitch		= metadata.pixelPitch;
	const double lensPitch		= metadata.lensPitch;
	const double rotationAngle	= metadata.rotationAngle;
	const double scaleFactorX	= metadata.scaleFactor[0];
	const double scaleFactorY	= metadata.scaleFactor[1];
	const double sensorOffsetX	= metadata.sensorOffset[0];
	const double sensorOffsetY	= metadata.sensorOffset[1];

	const Point2d imageCenter	= Point2d(hexagonalLensGrid.size()) * 0.5;
	const Point2d sensorOffset	= Point2d(sensorOffsetX, sensorOffsetY) * (1.0 / pixelPitch);
	const Point2d mlaCenter		= imageCenter + sensorOffset;

	// 2) determine size of the hexagonal grid and the rectilinear grid
	const double lensPitchInPixels		= lensPitch / pixelPitch;
	const double rowHeight				= lensPitchInPixels * cos(rotationAngle + M_PI / 6.0);
	const unsigned int lensImageLength	= (unsigned int) ceil(lensPitchInPixels);
	const Size lensImageSize			= Size(lensImageLength, lensImageLength);
	const unsigned int rowCount			= floor(hexagonalLensGrid.size().height / rowHeight) - 2;
	const unsigned int lensCountPerRow	= floor(hexagonalLensGrid.size().width / (lensPitchInPixels * cos(rotationAngle))) - 2;
	const Size rectifiedSize			= Size(lensCountPerRow * lensImageLength, rowCount * lensImageLength);

	// 3) prepare lens mask
	const int lensRadius		= (int) ceil(lensPitchInPixels / 2.0);
	Point2d lensCenter			= Point2d(lensRadius, lensRadius);
	const Scalar circleColor	= Scalar(1, 1, 1);
	Mat lensMask				= Mat::zeros(lensImageSize, CV_8UC1);
	circle(lensMask, lensCenter, lensRadius, circleColor, CV_FILLED);

	// 4) copy each lens' image from the hexagonal grid to the rectilinear grid
	Mat rectifiedLensGrid			= Mat::zeros(rectifiedSize, LightFieldFromLfpFile::IMAGE_TYPE);
	Rect srcRect, dstRect;
	Mat srcROI, dstROI;
	int  centeredRowIndex, centeredLensIndex;
	const double angleToNextRow			= rotationAngle + M_PI / 3.0;	// 60° to MLA axis
	const Point2d oneLensToTheRight		= Point2d(cos(rotationAngle), sin(rotationAngle)) * lensPitchInPixels;
	const Point2d toNextRow				= Point2d(cos(angleToNextRow), sin(angleToNextRow)) * lensPitchInPixels;;
	const Point2d fromCenterToCorner	= Point2d(1, 1) * -(lensPitchInPixels / 2.0);
	Point2d lensImageCorner;
	Point srcCorner, dstCorner;
	for (unsigned int rowIndex = 0; rowIndex < rowCount; rowIndex++)
	{
		for (unsigned int lensIndex = 0; lensIndex < lensCountPerRow; lensIndex++)
		{
			centeredRowIndex	= rowIndex - rowCount / 2;
			centeredLensIndex	= lensIndex - lensCountPerRow / 2;
			lensCenter	= (centeredLensIndex - floor(centeredRowIndex / 2.0)) * oneLensToTheRight +
				centeredRowIndex * toNextRow;
			lensCenter	= mlaCenter + Point2d(lensCenter.x * scaleFactorX,	// scaling
				lensCenter.y * scaleFactorY);
			lensImageCorner = lensCenter + fromCenterToCorner;
			srcCorner	= Point(round(lensImageCorner.x), round(lensImageCorner.y));

			dstCorner	= Point(lensIndex, rowIndex) * (int) lensImageLength;

			srcRect	= Rect(lensImageCorner, lensImageSize);
			dstRect	= Rect(dstCorner, lensImageSize);

			srcROI	= Mat(hexagonalLensGrid, srcRect);
			dstROI	= Mat(rectifiedLensGrid, dstRect);

			getRectSubPix(inputImage, lensImageSize, lensCenter, dstROI);
			//srcROI.copyTo(dstROI, lensMask);
			//srcROI.copyTo(dstROI);

			//rectangle(hexagonalLensGrid, srcRect, Scalar(255,0,0), 1);
			//rectangle(rectifiedLensGrid, dstRect, Scalar(255,0,0), 1);
			//circle(hexagonalLensGrid, Point(round(lensCenter.x), round(lensCenter.y)), 0, Scalar(255,0,0));
		}
	}

	return rectifiedLensGrid;
}


LightFieldFromLfpFile::LightFieldFromLfpFile(void)
{
}


LightFieldFromLfpFile::LightFieldFromLfpFile(const std::string& pathToFile)
{
	// load raw data
	this->loader	= LfpLoader(pathToFile);

	// calculate angular resolution of the light field
	const double lensPitch		= loader.lensPitch / loader.pixelPitch;
	const int lensPitchInPixels	= ceil(lensPitch);
	this->ANGULAR_RESOLUTION	= Size(lensPitchInPixels, lensPitchInPixels);

	// calculate spartial resolution of the light field
	const double xComponent			= lensPitch * cos(loader.rotationAngle);
	const unsigned int columnCount	= floor(this->loader.bayerImage.size().width / xComponent) - 2;

	const double yComponent		= lensPitch * cos(loader.rotationAngle + M_PI / 6.0);
	const unsigned int rowCount	= floor(this->loader.bayerImage.size().height / yComponent) - 2;

	this->SPARTIAL_RESOLUTION	= Size(columnCount, rowCount);

	// process raw image
	Mat rgbImage		= this->convertBayer2RGB(loader.bayerImage);
	Mat rectifiedImage	= this->rectifyLensGrid(rgbImage, loader);

	// scale luminance space to (image) data type
	Mat floatImage = adjustLuminanceSpace(rectifiedImage);

	this->rawImage	= floatImage;

	// generate all sub-aperture images
	size_t saImageCount = this->ANGULAR_RESOLUTION.width * this->ANGULAR_RESOLUTION.height;
	this->subapertureImages = vector<Mat>(saImageCount);
	int index = 0;
	for (int v = 0; v < this->ANGULAR_RESOLUTION.height; v++)
	{
		for (int u = 0; u < this->ANGULAR_RESOLUTION.width; u++)
		{
			this->subapertureImages[index] = this->generateSubapertureImage(u, v);
			index++;
		}
	}
}


LightFieldFromLfpFile::~LightFieldFromLfpFile(void)
{
	this->rawImage.release();
}


Vec3f LightFieldFromLfpFile::getLuminance(unsigned short x, unsigned short y, unsigned short u, unsigned short v)
{
	// handle coordinates outside the recorded lightfield
	const Point origin = Point(0, 0);
	const Rect validSpartialCoordinates = Rect(origin, this->SPARTIAL_RESOLUTION);
	// luminance outside the recorded spartial range is zero
	const Vec3f zeroLuminance = Vec3f(0, 0, 0);
	if (!validSpartialCoordinates.contains(Point(x, y)))
		return zeroLuminance;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	const Vec2f translationToOrigin = Vec2f(this->ANGULAR_RESOLUTION.width,
		this->ANGULAR_RESOLUTION.height) * -0.5;
	Vec2f angularVector = Vec2f(u, v);
	angularVector += translationToOrigin;	// center value range at (0, 0)
	double nrm = norm(angularVector);
	const double microLensRadiusInPixels = this->loader.lensPitch / (2.0 * this->loader.pixelPitch);
	if (nrm > microLensRadiusInPixels)
	{
		angularVector *= microLensRadiusInPixels / nrm;
		angularVector[0] = roundToZero(angularVector[0]);
		angularVector[1] = roundToZero(angularVector[1]);
		angularVector -= translationToOrigin;
		u = angularVector[0];
		v = angularVector[1];
	}

	Point pixelPosition = Point(x * this->ANGULAR_RESOLUTION.width + u,
		y * this->ANGULAR_RESOLUTION.height + v);
	return rawImage.at<Vec3f>(pixelPosition);
}


Vec3f LightFieldFromLfpFile::getSubpixelLuminance(unsigned short x, unsigned short y, unsigned short u, unsigned short v)
{
	// handle coordinates outside the recorded lightfield
	const Point origin = Point(0, 0);
	const Rect validSpartialCoordinates = Rect(origin, this->SPARTIAL_RESOLUTION);
	// luminance outside the recorded spartial range is zero
	const Vec3f zeroLuminance = Vec3f(0, 0, 0);
	if (!validSpartialCoordinates.contains(Point(x, y)))
		return zeroLuminance;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	const Vec2f translationToOrigin = Vec2f(this->ANGULAR_RESOLUTION.width,
		this->ANGULAR_RESOLUTION.height) * -0.5;
	Vec2f angularVector = Vec2f(u, v);
	angularVector += translationToOrigin;	// center value range at (0, 0)
	double nrm = norm(angularVector);
	const double microLensRadiusInPixels = this->loader.lensPitch / (2.0 * this->loader.pixelPitch);
	if (nrm > microLensRadiusInPixels)
	{
		angularVector *= microLensRadiusInPixels / nrm;
		angularVector -= translationToOrigin;
		u = angularVector[0];
		v = angularVector[1];
	}

	Mat singlePixel;
	const Size singlePixelSize = Size(1, 1);
	Point2f center = Point2f(x * this->ANGULAR_RESOLUTION.width + u,
		y * this->ANGULAR_RESOLUTION.height + v);
	getRectSubPix(rawImage, singlePixelSize, center, singlePixel);
	return singlePixel.at<Vec3f>(origin);
}


Vec3f LightFieldFromLfpFile::getLuminanceF(float x, float y, float u, float v)
{
	// handle coordinates outside the recorded lightfield
	const float halfWidth = this->SPARTIAL_RESOLUTION.width / 2.0;
	const float halfHeight = this->SPARTIAL_RESOLUTION.height / 2.0;
	// luminance outside the recorded spartial range is zero
	const Vec3f zeroLuminance = Vec3f(0, 0, 0);
	if (abs(x) > halfWidth || abs(y) > halfHeight)
		return zeroLuminance;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	Vec2f angularVector = Vec2f(u, v);
	double nrm = norm(angularVector);
	const double microLensRadiusInPixels = this->loader.lensPitch / (2.0 * this->loader.pixelPitch);
	if (nrm > microLensRadiusInPixels)
	{
		//angularVector *= microLensRadiusInPixels / nrm;
		normalize(angularVector, angularVector, microLensRadiusInPixels);
		return getLuminance(x, y, angularVector[0], angularVector[1]);
	}

	/*
	const unsigned int rawX = x * this->ANGULAR_RESOLUTION.width + u;
	const unsigned int rawY = y * this->ANGULAR_RESOLUTION.height + v;

	return this->rawImage.at<Vec3f>(Point(rawX, rawY));
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

	return singlePixel.at<Vec3f>(Point(0, 0));
}


Mat LightFieldFromLfpFile::generateSubapertureImage(const unsigned short u, const unsigned short v)
{
	Mat subapertureImage(this->SPARTIAL_RESOLUTION, CV_32FC3);

	for (int y = 0; y < this->SPARTIAL_RESOLUTION.height; y++)
		for (int x = 0; x < this->SPARTIAL_RESOLUTION.width; x++)
		{
			subapertureImage.at<Vec3f>(Point(x, y)) = this->getLuminance(x, y, u, v);
		}

	return subapertureImage;
}


Mat LightFieldFromLfpFile::getSubapertureImage(const unsigned short u, const unsigned short v)
{
	return this->subapertureImages[v * this->ANGULAR_RESOLUTION.width + u];
}


Mat LightFieldFromLfpFile::getSubapertureImageF(const double u, const double v)
{
	Mat upperLeftImage	= this->subapertureImages[floor(v) * this->ANGULAR_RESOLUTION.width + floor(u)];
	Mat lowerLeftImage	= this->subapertureImages[ceil(v) * this->ANGULAR_RESOLUTION.width + floor(u)];
	Mat upperRightImage	= this->subapertureImages[floor(v) * this->ANGULAR_RESOLUTION.width + ceil(u)];
	Mat lowerRightImage	= this->subapertureImages[ceil(v) * this->ANGULAR_RESOLUTION.width + ceil(u)];

	float lowerWeight	= v - floor(v);
	float upperWeight	= 1.0 - lowerWeight;
	float rightWeight	= u - floor(u);
	float leftWeight	= 1.0 - rightWeight;

	float upperLeftWeight	= upperWeight * leftWeight;
	float lowerLeftWeight	= lowerWeight * leftWeight;
	float upperRightWeight	= upperWeight * rightWeight;
	float lowerRightWeight	= lowerWeight * rightWeight;

	return upperLeftImage * upperLeftWeight + upperRightImage * upperRightWeight +
		lowerLeftImage * lowerLeftWeight + lowerRightImage * lowerRightWeight;
}


Mat LightFieldFromLfpFile::getRawImage()
{
	return this->rawImage;
}


double LightFieldFromLfpFile::getRawFocalLength()
{
	return this->loader.focalLength;
}
