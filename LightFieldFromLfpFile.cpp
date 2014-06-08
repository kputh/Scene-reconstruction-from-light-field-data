#define _USE_MATH_DEFINES	// for math constants in C++
#include <string>
#include <cmath>
#include <iostream>	// debug
#include <opencv2/imgproc/imgproc.hpp>
#include "LfpLoader.h"
#include "LightFieldFromLfpFile.h"


double round(double value)
{
	return (value < 0.0) ? ceil(value - 0.5) : floor(value + 0.5);;
}


Mat LightFieldFromLfpFile::convertBayer2RGB(const Mat bayerImage)
{
	Mat colorImage(bayerImage.size(), CV_16UC3);
	cvtColor(bayerImage, colorImage, CV_BayerBG2RGB);
	return colorImage;
}


Mat LightFieldFromLfpFile::rectifyLensGrid(const Mat hexagonalLensGrid, LfpLoader metadata)
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
	Mat rectifiedLensGrid			= Mat::zeros(rectifiedSize, hexagonalLensGrid.type());
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

			srcROI.copyTo(dstROI, lensMask);
			//srcROI.copyTo(dstROI);

			//rectangle(hexagonalLensGrid, srcRect, Scalar(255,0,0), 1);
			//rectangle(rectifiedLensGrid, dstRect, Scalar(255,0,0), 1);
			//circle(hexagonalLensGrid, Point(round(lensCenter.x), round(lensCenter.y)), 0, Scalar(255,0,0));
		}
	}

	//return hexagonalLensGrid;
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

	this->rawImage	= rectifiedImage;
}


LightFieldFromLfpFile::~LightFieldFromLfpFile(void)
{
	this->rawImage.release();
}


Vec3s LightFieldFromLfpFile::getLuminance(const unsigned short x, const unsigned short y, const unsigned short u, const unsigned short v)
{
	// handle coordinates outside the recorded lightfield
	const Point origin = Point(0, 0);
	const Rect validSpartialCoordinates = Rect(origin, this->SPARTIAL_RESOLUTION);
	const Rect validAngularCoordinates = Rect(origin, this->ANGULAR_RESOLUTION);
	if (!validSpartialCoordinates.contains(Point(x, y)) ||
		!validAngularCoordinates.contains(Point(u, v)))
		return Vec3s(0, 0, 0);

	const unsigned int rawX = x * this->ANGULAR_RESOLUTION.width + u;
	const unsigned int rawY = y * this->ANGULAR_RESOLUTION.height + v;

	return this->rawImage.at<Vec3s>(Point(rawX, rawY));
	//TODO use getRectSubPix() instead
}


Mat LightFieldFromLfpFile::getSubapertureImage(const unsigned short u, const unsigned short v)
{
	Mat subapertureImage(this->SPARTIAL_RESOLUTION, this->rawImage.type());

	for (int y = 0; y < this->SPARTIAL_RESOLUTION.height; y++)
		for (int x = 0; x < this->SPARTIAL_RESOLUTION.width; x++)
		{
			subapertureImage.at<Vec3s>(Point(x, y)) = this->getLuminance(x, y, u, v);
		}

	return subapertureImage;
}


Mat LightFieldFromLfpFile::getAllSubaperturesInOneImage()
{
	Mat allSAImages(this->rawImage.size(), CV_16UC3);

	unsigned short x, y, u, v;
	unsigned int newX, newY, effectiveX;
	const unsigned int ODD_ROW_LENS_OFFSET = this->SPARTIAL_RESOLUTION.width / 2;
	for (int rawY = 0; rawY < 3280; rawY++)
	{
		y = rawY / this->ANGULAR_RESOLUTION.height;
		v = rawY % this->ANGULAR_RESOLUTION.height;
		newY = v * this->SPARTIAL_RESOLUTION.height + y;

		for (int rawX = 0; rawX < 3280; rawX++)
		{
			effectiveX = (y % 2 == 0) ? rawX : rawX - ODD_ROW_LENS_OFFSET;
			x = effectiveX / this->ANGULAR_RESOLUTION.width;
			u = effectiveX % this->ANGULAR_RESOLUTION.width;
			newX = u * this->SPARTIAL_RESOLUTION.width + x;

			allSAImages.at<Vec3s>(newX, newY) = this->rawImage.at<Vec3s>(rawX, rawY);
		}
	}

	return allSAImages;
}


Mat LightFieldFromLfpFile::getImage(const double focalLength)
{
	const double F		= this->loader.focalLength;	// focal length of the raw image
	const double alpha	= focalLength / F;

	const double weight = 1.0 - 1.0 / alpha;
	const Size dilatedSize = Size(this->SPARTIAL_RESOLUTION.width * alpha,
		this->SPARTIAL_RESOLUTION.height * alpha);
	const Size imageSize = Size(dilatedSize.width + this->ANGULAR_RESOLUTION.width * abs(weight),
		dilatedSize.height + this->ANGULAR_RESOLUTION.height * abs(weight));
	const int imageType = CV_MAKETYPE(CV_32F, this->rawImage.channels());
	Mat image = Mat::zeros(imageSize, imageType);

	Mat subapertureImage, resizedSAImage, dstROI;
	Vec2d translation, dstCorner;
	const Vec2d angularCorrection = Vec2d(this->ANGULAR_RESOLUTION.width, this->ANGULAR_RESOLUTION.height) * 0.5;
	const Vec2d dstCenter = Vec2d(image.size().width, image.size().height) * 0.5;
	Rect dstRect;
	const Vec2d fromCenterToCorner = Vec2d(dilatedSize.width, dilatedSize.height) * -0.5;
	const int interpolationMethod = (alpha < 0.0) ? CV_INTER_AREA : CV_INTER_CUBIC;

	for(int u = 0; u < this->ANGULAR_RESOLUTION.width; u++)
	{
		for(int v = 0; v < this->ANGULAR_RESOLUTION.height; v++)
		{
			subapertureImage = this->getSubapertureImage(u, v);

			resize(subapertureImage, resizedSAImage, dilatedSize, 0, 0, interpolationMethod);

			translation	= (Vec2d(u, v) - angularCorrection) * weight;
			dstCorner	= dstCenter + translation + fromCenterToCorner;
			dstRect		= Rect(Point(round(dstCorner[0]), round(dstCorner[1])), dilatedSize);
			dstROI		= Mat(image, dstRect);

			add(resizedSAImage, dstROI, dstROI, noArray(), dstROI.type());
		}
	}

	//image *= 1.0 / (alpha * alpha * F * F);	// part of the formular, not required because of normalization

	// adjust luminance
	normalize(image, image, 0.0, 1.0, NORM_MINMAX);

	return image;
}


Mat LightFieldFromLfpFile::getRawImage()
{
	return this->rawImage;
}
