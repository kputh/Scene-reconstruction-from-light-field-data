#include <string>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "LfpLoader.h"
#include "LightFieldFromLfpFile.h"


Mat LightFieldFromLfpFile::convertBayer2RGB(const Mat bayerImage)
{
	Mat colorImage(bayerImage.size(), CV_16UC3);
	cvtColor(bayerImage, colorImage, CV_BayerBG2RGB);
	return colorImage;
}


Mat LightFieldFromLfpFile::rectifyLensGrid(const Mat hexagonalLensGrid)
{
	// 1) determine dimensions of the hexagonal grid and the rectilinear grid
	const Size lensSize					= LightFieldFromLfpFile::ANGULAR_RESOLUTION;
	const unsigned int rowCount			= hexagonalLensGrid.size().height / lensSize.height - 1; // TODO -1?
	const unsigned int lensCountPerRow	= hexagonalLensGrid.size().width / lensSize.width - 1;	// TODO -1?
	const Size rowSize					= Size(lensCountPerRow * lensSize.width, lensSize.height);
	const Size rectifiedSize			= Size(lensCountPerRow * lensSize.width, rowCount * lensSize.height);

	// 2) prepare row mask
	const Point toNextLensInRow			= Point(lensSize.width, 0);
	const unsigned short lensRadius		= lensSize.width / 2;
	const Scalar circleColor			= Scalar(1,1,1);
	const int circleFill				= -1;
	Mat rowMask							= Mat::zeros(rowSize, CV_8UC1);
	Point center						= Point(lensRadius, lensRadius);
	for (unsigned int lensIndex = 0; lensIndex < lensCountPerRow; lensIndex++)
	{
		// mark area of current lens' image
		circle(rowMask, center, lensRadius, circleColor, circleFill);

		// move one lens to the right
		center += toNextLensInRow;
	}

	// [unwarp image of hexagonal grid]
	const double rotationAngle	= 0.002145454753190279; // TODO
	const Point rotationCenter	= Point(5, 3);
	const double rotationScale	= 1.0;
    const Mat rotation			= getRotationMatrix2D(rotationCenter, rotationAngle, rotationScale);
	Mat alignedGrid;
	warpAffine(hexagonalLensGrid, alignedGrid, rotation, hexagonalLensGrid.size());

	// 3) copy each row from the hexagonal grid to the rectilinear grid
	Mat rectifiedLensGrid = Mat::zeros(rectifiedSize, hexagonalLensGrid.type());
	Rect srcRect, dstRect;
	Mat srcROI, dstROI;
	const float srcBaseY			= 3.0;
	const float srcIncrementY		= 8.6;
	const unsigned int srcOddRowX	= 0;
	const unsigned int srcEvenRowX	= 5;
	unsigned int srcX, srcY, dstY;
	const unsigned int dstX			= 0;
	for (unsigned int rowIndex = 0; rowIndex < rowCount; rowIndex++)
	{
		srcX	= (rowIndex % 2 == 0) ? srcEvenRowX : srcOddRowX;
		srcY	= floor((srcBaseY + rowIndex * srcIncrementY) + 0.5);
		srcRect	= Rect(srcX, srcY, rowSize.width, rowSize.height);
		dstY	= rowIndex * lensSize.height;
		dstRect	= Rect(dstX, dstY, rowSize.width, rowSize.height);
		srcROI	= Mat(hexagonalLensGrid, srcRect);
		dstROI	= Mat(rectifiedLensGrid, dstRect);
		//dstROI.setTo(srcROI, rowMask);
		srcROI.copyTo(dstROI);
		//srcROI.copyTo(dstROI, rowMask);
		//dstROI = srcROI * rowMask;
	}

	//return srcROI;
	//return hexagonalLensGrid;
	//return rowMask;
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
	const bool isOddRow = (y & 1 == 1);
	const unsigned int columnOffset = isOddRow ? (ANGULAR_RESOLUTION.width / 2.) : 0;
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
