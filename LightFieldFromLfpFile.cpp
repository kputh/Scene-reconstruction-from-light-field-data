#define _USE_MATH_DEFINES	// for math constants in C++
#include <string>
#include <cmath>
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


Mat LightFieldFromLfpFile::rectifyLensGrid(const Mat hexagonalLensGrid)
{
	// 1) read LFP metadata
	// TODO
	const double pixelPitch		= 0.0000013999999761581417;
	const double lensPitch		= 0.00001389859962463379;
	const double rotationAngle	= 0.002145454753190279;
	const double scaleFactorX	= 1.0;
	const double scaleFactorY	= 1.0014984607696533;
	const double sensorOffsetX	= 0.0000018176757097244258;
	const double sensorOffsetY	= -0.0000040150876045227051;

	const Point2d imageCenter	= Point2d(hexagonalLensGrid.size()) * 0.5;
	const Point2d sensorOffset	= Point2d(sensorOffsetX, sensorOffsetY) * (1.0 / pixelPitch);
	const Point2d mlaCenter		= imageCenter + sensorOffset;

	// 2) determine size of the hexagonal grid and the rectilinear grid
	const double lensPitchInPixels		= lensPitch / pixelPitch;
	const double rowHeight				= lensPitchInPixels * cos(rotationAngle + M_PI / 6.0);
	const unsigned int lensImageLength	= (unsigned int) ceil(lensPitchInPixels);
	const Size lensImageSize			= Size(lensImageLength, lensImageLength);
	const unsigned int rowCount			= floor(hexagonalLensGrid.size().height / rowHeight) - 2;
	const unsigned int lensCountPerRow	= floor(hexagonalLensGrid.size().width / lensPitchInPixels) - 2;
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

	
const cv::Size LightFieldFromLfpFile::SPARTIAL_RESOLUTION = Size(328, 328);
const cv::Size LightFieldFromLfpFile::ANGULAR_RESOLUTION = Size(10, 10);


LightFieldFromLfpFile::LightFieldFromLfpFile(void)
{
}


LightFieldFromLfpFile::LightFieldFromLfpFile(const std::string& pathToFile)
{
	Mat bayerImage = LfpLoader::loadAsBayer(pathToFile);
	Mat rgbImage = this->convertBayer2RGB(bayerImage);
	Mat rectifiedImage = this->rectifyLensGrid(rgbImage);

	this->rawImage = rectifiedImage;
}


LightFieldFromLfpFile::~LightFieldFromLfpFile(void)
{
	this->rawImage.release();
}


Vec3s LightFieldFromLfpFile::getLuminance(const unsigned short x, const unsigned short y, const unsigned short u, const unsigned short v)
{
	// ToDo odd rows
	const bool isOddRow = ((y & 1) == 1);
	const unsigned int columnOffset = isOddRow ? (ANGULAR_RESOLUTION.width / 2) : 0;
	const unsigned int rawX = x * this->ANGULAR_RESOLUTION.width + columnOffset + u;
	const unsigned int rawY = y * this->ANGULAR_RESOLUTION.height + v;

	return this->rawImage.at<Vec3s>(rawX, rawY);
}


Mat LightFieldFromLfpFile::getSubapertureImage(const unsigned short u, const unsigned short v)
{
	Mat subapertureImage(this->SPARTIAL_RESOLUTION, CV_16UC3);

	for (int y = 0; y < this->SPARTIAL_RESOLUTION.height; y++)
		for (int x = 0; x < this->SPARTIAL_RESOLUTION.width - 1; x++)
		{
			subapertureImage.at<Vec3s>(x, y) = this->getLuminance(x, y, u, v);
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

Mat LightFieldFromLfpFile::getRawImage()
{
	return this->rawImage;
}
