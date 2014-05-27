#include <string>
#include "LfpLoader.h"
#include "LightFieldFromLfpFile.h"


const cv::Size LightFieldFromLfpFile::SPARTIAL_RESOLUTION = Size(328, 328);
const cv::Size LightFieldFromLfpFile::ANGULAR_RESOLUTION = Size(10, 10);


LightFieldFromLfpFile::LightFieldFromLfpFile(void)
{
}


LightFieldFromLfpFile::LightFieldFromLfpFile(const std::string& pathToFile)
{
	this->rawLfp = LfpLoader::loadAsRGB(pathToFile);
}


LightFieldFromLfpFile::~LightFieldFromLfpFile(void)
{
	this->rawLfp.release();
}


Vec3s LightFieldFromLfpFile::getLuminance(const unsigned short x, const unsigned short y, const unsigned short u, const unsigned short v)
{
	// ToDo odd rows
	const bool isOddRow = (y & 1 == 1);
	const unsigned int columnOffset = isOddRow ? (ANGULAR_RESOLUTION.width / 2.) : 0;
	const unsigned int rawX = x * this->ANGULAR_RESOLUTION.width + columnOffset + u;
	const unsigned int rawY = y * this->ANGULAR_RESOLUTION.height + v;

	return this->rawLfp.at<Vec3s>(rawX, rawY);
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

Mat getAllSubaperturesInOneImage()
{
	Mat allSAImages(this->rawLfp.size(), CV_16UC3);
	for (int u = 0; u < this->ANGULAR_RESOLUTION.width; u++)
	{
		for (int v = 0; v < this->ANGULAR_RESOLUTION.width; v++)
		{
		}
	}

	unsigned short x, y, u, v;
	Mat regionOfInterest;
	Mat sai;
	for (int rawX = 0; rawX < 3280; rawX++)
		for (int rawY = 0; rawY < 3280; rawY++)
		{
			regionOfInterest = this->rawLfp.colRange(1,2).rowRange(3,4);
			sai.copyTo(regionOfInterest):
			/*
			x = rawX / this->ANGULAR_RESOLUTION.width; // ToDo
			y = rawY / this->ANGULAR_RESOLUTION.height;
			u = rawX % this->ANGULAR_RESOLUTION.width;	// ToDo
			v = rawY % this->ANGULAR_RESOLUTION.height;
			allSAImages.at<Vec3s>(rawY, rawY) = this->getLuminance(x, y, u, v);
			*/
		}

}

Mat LightFieldFromLfpFile::getRawImage()
{
	return this->rawLfp;
}
