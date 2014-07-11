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


Mat LightFieldPicture::demosaicImage(const Mat& bayerImage)
{
	Mat demosaicedImage(bayerImage.size(), CV_16UC3);
	cvtColor(bayerImage, demosaicedImage, CV_BayerBG2RGB);
	return demosaicedImage;
}


void LightFieldPicture::rectifyLensGrid(Mat& hexagonalLensGrid,
	const LfpLoader& metadata)
{
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
	Mat rectifiedLensGrid = Mat(rectifiedSize, IMAGE_TYPE, Scalar(0));
	Rect srcRect, dstRect;
	Mat srcROI, dstROI;
	int  centeredRowIndex, centeredLensIndex;
	const double angleToNextRow = rotationAngle + M_PI / 3.0;	// 60° to MLA axis
	const Point2d oneLensToTheRight = Point2d(cos(rotationAngle), sin(rotationAngle)) * lensPitchInPixels;
	const Point2d toNextRow = Point2d(cos(angleToNextRow), sin(angleToNextRow)) * lensPitchInPixels;;
	const Point2d fromCenterToCorner	= Point2d(1, 1) * -(lensPitchInPixels / 2.0);
	Point2d lensImageCorner;
	Point srcCorner, dstCorner;
	unsigned int rowIndex, lensIndex;
	for (rowIndex = 0; rowIndex < rowCount; rowIndex++)
	{
		for (lensIndex = 0; lensIndex < lensCountPerRow; lensIndex++)
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

			getRectSubPix(hexagonalLensGrid, lensImageSize, lensCenter, dstROI);
			//srcROI.copyTo(dstROI, lensMaskOcl);
			//srcROI.copyTo(dstROI);

			//rectangle(hexagonalLensGrid, srcRect, Scalar(255,0,0), 1);
			//rectangle(rectifiedLensGrid, dstRect, Scalar(255,0,0), 1);
			//circle(hexagonalLensGrid, Point(round(lensCenter.x), round(lensCenter.y)), 0, Scalar(255,0,0));
		}
	}

	hexagonalLensGrid = rectifiedLensGrid;
}


oclMat LightFieldPicture::extractSubapertureImageAtlas(
	const Mat& hexagonalLensGrid, const LfpLoader& metadata)
{
	// 1) read LFP metadata
	const double pixelPitch		= metadata.pixelPitch;
	const double lensPitch		= metadata.lensPitch;
	const double rotationAngle	= 0;//-metadata.rotationAngle;
	const double scaleFactorX	= metadata.scaleFactor[0];
	const double scaleFactorY	= metadata.scaleFactor[1];
	const double sensorOffsetX	= metadata.sensorOffset[0];
	const double sensorOffsetY	= metadata.sensorOffset[1];

	const Size srcSize			= hexagonalLensGrid.size();
	const Vec2f sensorCenter	= Vec2f(srcSize.width, srcSize.height) * 0.5;
	const Vec2f mlaOffset		= Vec2f(sensorOffsetX, sensorOffsetY) / pixelPitch;
	const Vec2f mlaCenter		= sensorCenter + mlaOffset;

	// 2) determine size of the hexagonal grid and the rectilinear grid
	const double lensPitchInPixels		= lensPitch / pixelPitch;

	const Size correctedSrcSize = RotatedRect(mlaCenter, srcSize, rotationAngle)
		.boundingRect().size();
	const double lensWidth = lensPitchInPixels * scaleFactorX;
	const double rowHeight = lensPitchInPixels * scaleFactorY * cos(M_PI / 6.0);
	const int colCount = floor(correctedSrcSize.width / lensWidth) - 1;
	const int rowCount = floor(correctedSrcSize.height / rowHeight) - 1;

	const int lensImageLength = ceil(lensPitchInPixels);
	const int dstWidth = colCount * lensImageLength;
	const int dstHeight = rowCount * lensImageLength;

	const double angleToNextRow = M_PI / 3.0 + rotationAngle;	// 60° to MLA axis
	const Vec2f nextLens = Vec2f(cos(rotationAngle), sin(rotationAngle))
		* lensPitchInPixels;
	const Vec2f nextRow = Vec2f(cos(angleToNextRow), sin(angleToNextRow))
		* lensPitchInPixels;

	const int u0 = lensImageLength / 2;
	const int v0 = lensImageLength / 2;
	const int s0 = colCount / 2;
	const int t0 = rowCount / 2 - 1;

	Mat map = Mat(dstHeight, dstWidth, CV_32FC2);
	int x, y, s, t, u, v;
	for (y = 0; y < map.rows; y++)
	{
		v = y / rowCount - v0;
		t = y % rowCount - t0;

		for (x = 0; x < map.cols; x++)
		{
			u = x / colCount - u0;
			s = x % colCount - s0;

			map.at<Vec2f>(y, x) = floor(s - t / 2.) * nextLens + t * nextRow
				+ Vec2f(u, v) + mlaCenter;
		}
	}

	//cout << "map = " << map << endl;
	oclMat subapertureImageAtlas;
	ocl::remap(oclMat(hexagonalLensGrid), subapertureImageAtlas, oclMat(map), oclMat(),
		CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,65535));	// TODO use all parameters

	return subapertureImageAtlas;
}


LightFieldPicture::LightFieldPicture(void)
{
}


LightFieldPicture::LightFieldPicture(const std::string& pathToFile)
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

	this->validSpartialCoordinates = Rect(IMAGE_ORIGIN, SPARTIAL_RESOLUTION);
	this->microLensRadiusInPixels = loader.lensPitch / (2.0 * loader.pixelPitch);
	this->fromLensCenterToOrigin = Vec2f(ANGULAR_RESOLUTION.width,
		ANGULAR_RESOLUTION.height) * -0.5;

	// process raw image
	Mat demosaicedImage	= demosaicImage(loader.bayerImage);
	demosaicedImage.copyTo(this->rawImage);

	this->subapertureImageAtlas = extractSubapertureImageAtlas(demosaicedImage, loader);
	/*
	cvtColor(demosaicedImage, demosaicedImage, CV_RGB2GRAY);
	demosaicedImage.convertTo(demosaicedImage, CV_32FC1, 1.0 / 65535.0);
	rectifyLensGrid(demosaicedImage, loader); // TODO abschaffen?
	demosaicedImage.copyTo(this->processesImage);

	// generate all sub-aperture images
	size_t saImageCount = ANGULAR_RESOLUTION.area();
	subapertureImages = vector<oclMat>(saImageCount);
	int u, v, index = 0;
	for (v = 0; v < ANGULAR_RESOLUTION.height; v++)
	{
		for (u = 0; u < ANGULAR_RESOLUTION.width; u++)
		{
			subapertureImages[index] = generateSubapertureImage(u, v);
			index++;
		}
	}
	*/
}


LightFieldPicture::~LightFieldPicture(void)
{
	this->rawImage.release();
}


LightFieldPicture::luminanceType LightFieldPicture::getLuminance(
	unsigned short x, unsigned short y, unsigned short u, unsigned short v) const
{
	// luminance outside the recorded spartial range is zero
	if (!validSpartialCoordinates.contains(Point(x, y)))
		return ZERO_LUMINANCE;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	Vec2f angularVector = Vec2f(u, v);
	angularVector += fromLensCenterToOrigin;	// center value range at (0, 0)
	double nrm = norm(angularVector);
	if (nrm > microLensRadiusInPixels)
	{
		angularVector *= microLensRadiusInPixels / nrm;
		angularVector[0] = roundToZero(angularVector[0]);
		angularVector[1] = roundToZero(angularVector[1]);
		angularVector -= fromLensCenterToOrigin;
		u = angularVector[0];
		v = angularVector[1];
	}

	Point pixelPosition = Point(x * this->ANGULAR_RESOLUTION.width + u,
		y * this->ANGULAR_RESOLUTION.height + v);
	return processesImage.at<luminanceType>(pixelPosition);
}


LightFieldPicture::luminanceType LightFieldPicture::getSubpixelLuminance(
	unsigned short x, unsigned short y, unsigned short u, unsigned short v) const
{
	// luminance outside the recorded spartial range is zero
	if (!validSpartialCoordinates.contains(Point(x, y)))
		return ZERO_LUMINANCE;

	// luminance outside the recorded angular range is clamped to the closest valid ray
	Vec2f angularVector = Vec2f(u, v);
	angularVector += fromLensCenterToOrigin;	// center value range at (0, 0)
	double nrm = norm(angularVector);
	if (nrm > microLensRadiusInPixels)
	{
		angularVector *= microLensRadiusInPixels / nrm;
		angularVector -= fromLensCenterToOrigin;
		u = angularVector[0];
		v = angularVector[1];
	}

	Mat singlePixel;
	const Size singlePixelSize = Size(1, 1);
	Point2f center = Point2f(x * this->ANGULAR_RESOLUTION.width + u,
		y * this->ANGULAR_RESOLUTION.height + v);
	getRectSubPix(processesImage, singlePixelSize, center, singlePixel);
	return singlePixel.at<luminanceType>(IMAGE_ORIGIN);
}


LightFieldPicture::luminanceType LightFieldPicture::getLuminanceF(
	float x, float y, float u, float v) const
{
	// handle coordinates outside the recorded lightfield
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
		return getLuminance(x, y, angularVector[0], angularVector[1]);
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
	getRectSubPix(processesImage, singlePixelSize, Point2f(position), singlePixel);

	return singlePixel.at<luminanceType>(Point(0, 0));
}


oclMat LightFieldPicture::generateSubapertureImage(const unsigned short u,
	const unsigned short v) const
{
	Mat_<luminanceType> subapertureImage(this->SPARTIAL_RESOLUTION, IMAGE_TYPE);

	int x, y;
	for (y = 0; y < this->SPARTIAL_RESOLUTION.height; y++)
		for (x = 0; x < this->SPARTIAL_RESOLUTION.width; x++)
		{
			subapertureImage.at<luminanceType>(y, x) =
				this->getLuminance(x, y, u, v);
		}

	return oclMat(subapertureImage);
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
