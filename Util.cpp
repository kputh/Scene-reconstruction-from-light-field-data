#define _USE_MATH_DEFINES	// for math constants in C++

#include <cmath>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Util.h"
#include "ImageRenderer3.h"

double round(double value)
{
	return (value < 0.0) ? ceil(value - 0.5) : floor(value + 0.5);
}

double roundTo(double value, double target)
{
	return (value < target) ? ceil(value) : floor(value);
}

double roundToZero(double value)
{
	return (value < 0.0) ? ceil(value) : floor(value);
}

Mat adjustLuminanceSpace(const Mat image)
{
	Mat floatImage;
	double minValue, maxValue, scaleFactor, offset;
	minMaxLoc(image, &minValue, &maxValue);
	scaleFactor = 1.0 / (maxValue - minValue);
	offset = - minValue * scaleFactor;
	image.convertTo(floatImage, CV_32F, scaleFactor, offset);

	return floatImage;
}

void saveImageToPNGFile(string fileName, Mat image)
{
	const double maxValue16bit = 65535;
	double minValue, maxValue, scaleFactor;
	minMaxLoc(image, &minValue, &maxValue);
	scaleFactor = maxValue16bit / (maxValue - minValue);
	Mat writableMat;
	image.convertTo(writableMat, CV_16U, scaleFactor, -minValue * scaleFactor);

	imwrite(fileName, writableMat);
	cout << "Image saved as file " << fileName << "." << endl;
}

void saveImageArc(LightFieldPicture lightfield, string sourceFileName, int imageCount)
{
	float angle, x, y;
	float radius = 4;
	float f = 0.0068200001716613766;

	ImageRenderer3 renderer = ImageRenderer3();
	renderer.setFocalLength(f);
	renderer.setLightfield(lightfield);

	sourceFileName.erase(sourceFileName.end() - 4, sourceFileName.end());
	sourceFileName.append("_");
	string fileExtension = string(".png");
	string imageFileName;
	Mat image;
	for (int i = 0; i < imageCount; i++)
	{
		angle = M_PI / 4.0 * (float) i;
		x = cos(angle) * radius;
		y = sin (angle) * radius;
		renderer.setPinholePosition(Vec2i(round(x), round(y)));
		image = renderer.renderImage();
		imageFileName = sourceFileName + to_string((long double)i) + fileExtension;
		saveImageToPNGFile(imageFileName, image);
	}
	cout << "image arc saved" << endl;
}


Mat appendRayCountingChannel(Mat image)
{
	int compositeImageType = CV_MAKETYPE(image.depth(), image.channels() + 1);
	int counterImageType = CV_MAKETYPE(image.depth(), 1);
	Mat rayCount = Mat::ones(image.size(), counterImageType);
	Mat compositeImage = Mat(image.size(), compositeImageType);

	Mat in[] = { image, rayCount };
	Mat out[] = { compositeImage };
	int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
	mixChannels( in, 2, out, 1, from_to, 4);

	return compositeImage;
}


Mat normalizeByRayCount(Mat image)
{
	int imageType = CV_MAKETYPE(image.depth(), image.channels() - 1);
	Mat rayCount = Mat(image.size(), imageType);
	Mat normalizedImage = Mat(image.size(), imageType);

	Mat in[] = { image };
	Mat out[] = { normalizedImage, rayCount };
	int from_to[] = { 0,0, 1,1, 2,2, 3,3, 3,4, 3,5 };
	mixChannels( in, 1, out, 2, from_to, 6);

	divide(normalizedImage, rayCount, normalizedImage);

	return normalizedImage;
}