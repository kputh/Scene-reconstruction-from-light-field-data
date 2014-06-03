#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "LfpLoader.h"
#include "LightFieldFromLfpFile.h"


double round(double value)
{
	return floor(value + 0.5);
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
	const double rotationAngle	= 0.122925502;
	const double scaleFactorX	= 1.0;
	const double scaleFactorY	= 1.0014984607696533;

	// 2) align grid to image borders
	const Point rotationCenter	= Point(hexagonalLensGrid.size().width / 2, hexagonalLensGrid.size().height / 2);
	const double rotationScale	= 1.0;
    const Mat rotation			= getRotationMatrix2D(rotationCenter, rotationAngle, rotationScale);
	Mat rotatedGrid, alignedGrid;
	warpAffine(hexagonalLensGrid, rotatedGrid, rotation, hexagonalLensGrid.size());
	resize(rotatedGrid, alignedGrid, Size(), 1.0 / scaleFactorX, 1.0 / scaleFactorY);

	// 3) determine dimensions of the hexagonal grid and the rectilinear grid
	// TODO pixel pitch und lens pitch aus Metadaten des LFP auslesen
	const double lensPitchInPixels		= lensPitch / pixelPitch;
	const unsigned int lensImageLength	= (unsigned int) ceil(lensPitchInPixels);
	const unsigned int rowCount			= (unsigned int)((double) hexagonalLensGrid.size().height / lensPitchInPixels) - 2; // TODO -1?
	const unsigned int lensCountPerRow	= (unsigned int)((double) hexagonalLensGrid.size().width / lensPitchInPixels) - 2; // TODO -1?
	//const Size rowSize					= Size(lensCountPerRow * lensImageLength, lensImageLength);
	const Size rectifiedSize			= Size(lensCountPerRow * lensImageLength, rowCount * lensImageLength);

	// 4) prepare row mask
	const int lensRadius		= (int) ceil(lensPitchInPixels / 2.0);
	const Point lensCenter		= Point(lensRadius, lensRadius);
	const Scalar circleColor	= Scalar(1, 1, 1);
	const int circleFill		= -1;
	Mat lensMask				= Mat::zeros(lensImageLength, lensImageLength, CV_8UC1);
	circle(lensMask, lensCenter, lensRadius, circleColor, circleFill);

	// 5) copy each row from the hexagonal grid to the rectilinear grid
	Mat rectifiedLensGrid			= Mat::zeros(rectifiedSize, hexagonalLensGrid.type());
	Rect srcRect, dstRect;
	Mat srcROI, dstROI;
	const double PI					= 3.14159265359;
	const double srcBaseX			= 7.0;
	const double srcBaseY			= 6.0;
	const double srcIncrementY		= cos(PI / 6.0) * lensPitchInPixels; // Abstand übereinander liegender Linsen in Pixeln
	unsigned int srcX, srcY, dstX, dstY;
	for (unsigned int rowIndex = 0; rowIndex < rowCount; rowIndex++)
		for (unsigned int lensIndex = 0; lensIndex < lensCountPerRow; lensIndex++)
		{
			srcX	= (rowIndex % 2 == 0) ?
				(unsigned int) round(srcBaseX + ((double) lensIndex + 0.5) * lensPitchInPixels) :
				(unsigned int) round(srcBaseX + (double) lensIndex * lensPitchInPixels);
			srcY	= (unsigned int) round(srcBaseY + (double) rowIndex * srcIncrementY);
			dstX	= lensIndex * lensImageLength;
			dstY	= rowIndex * lensImageLength;
			// TODO compare to getRectSubPix()
			srcRect	= Rect(srcX, srcY, lensImageLength, lensImageLength);
			dstRect	= Rect(dstX, dstY, lensImageLength, lensImageLength);
			srcROI	= Mat(alignedGrid, srcRect);
			dstROI	= Mat(rectifiedLensGrid, dstRect);
			srcROI.copyTo(dstROI, lensMask);
		}

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
