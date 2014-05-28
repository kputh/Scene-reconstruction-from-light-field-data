#include <string>
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
	// 1) isolate one source area for odd and even rows each. Both areas have
	// the size of the output image. The lens' images are at their final
	// positions.
	// 2) create one mask for the lens' images for odd and even rows each
	// 3) paste the lens' images into a new, empty image.

	const unsigned int rowCount = hexagonalLensGrid.size().height / LightFieldFromLfpFile::ANGULAR_RESOLUTION.height; // TODO -1?
	const unsigned int lensCountPerRow = hexagonalLensGrid.size().width / LightFieldFromLfpFile::ANGULAR_RESOLUTION.width;	// TODO -1?
	const Size rowSize = Size(lensCountPerRow * LightFieldFromLfpFile::ANGULAR_RESOLUTION.width, LightFieldFromLfpFile::ANGULAR_RESOLUTION.height);

	Rect oddArea = Rect();	// TODO
	Rect evenArea = Rect();	// TODO
	Mat oddCut = Mat(hexagonalLensGrid, oddArea);
	Mat evenCut = Mat(hexagonalLensGrid, evenArea);

	Mat *currentMask;
	Mat oddMask		= Mat::zeros(hexagonalLensGrid.size(), CV_8UC1);
	Mat evenMask	= Mat::zeros(hexagonalLensGrid.size(), CV_8UC1);
	Point center = Point(5, 5);
	const Point toNextLensInRow = Point(10, 0);
	const unsigned short radius = 5;
	for (unsigned int rowIndex; rowIndex < rowCount; rowIndex++)
	{
		currentMask = (rowIndex % 2 == 0) ? evenMask : oddMask;
		// TODO center
		center = Point(0,0);
		for (unsigned int lensIndex; lensIndex < lensCountPerRow; lensIndex++)
		{
			center += toNextLensInRow;
			circle(currentMask, center, radius, Scalar(255,255,255), -1);
		}
	}

	Mat rectifiedLensGrid = Mat::zeros(evenCut.size(), hexagonalLensGrid.type());
	rectifiedLensGrid.setTo(oddCut, oddMask);
	rectifiedLensGrid.setTo(evenCut, evenMask);

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
