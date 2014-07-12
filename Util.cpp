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

Vec2d round(Vec2d vector)
{
	for(int i = 0; i < vector.rows; i++)
		vector[i] = round(vector[i]);

	return vector;
}

double roundTo(double value, double target)
{
	return (value < target) ? ceil(value) : floor(value);
}

double roundToZero(double value)
{
	return (value < 0.0) ? ceil(value) : floor(value);
}

Vec2d roundToZero(Vec2d vector)
{
	for(int i = 0; i < vector.rows; i++)
		vector[i] = roundToZero(vector[i]);

	return vector;
}

void adjustLuminanceSpace(Mat& image)	// ist das nicht identisch zu CV::normalize()?
{
	double minValue, maxValue, scaleFactor, offset;
	minMaxLoc(image, &minValue, &maxValue);
	scaleFactor = 1.0 / (maxValue - minValue);
	offset = - minValue * scaleFactor;
	image.convertTo(image, CV_32F, scaleFactor, offset);
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
	float alpha = 1.0;

	ImageRenderer3 renderer = ImageRenderer3();
	renderer.setAlpha(alpha);
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


void appendRayCountingChannel(Mat& image)
{
	vector<Mat> channels;
	split(image, channels);

	Mat rayCountChannel = Mat::ones(image.size(), channels[0].type());
	channels.push_back(rayCountChannel);
	
	merge(channels, image);
}


void normalizeByRayCount(Mat& image)
{
	vector<Mat> channels;
	split(image, channels);

	int rayCountChannelIndex = channels.size() - 1;
	for (int i = 0; i < rayCountChannelIndex; i++)
	{
		divide(channels[i], channels[rayCountChannelIndex], channels[i]);
	}
	channels.pop_back();

	merge(channels, image);
}


void appendRayCountingChannel(oclMat& image)
{
	vector<oclMat> channels;
	ocl::split(image, channels);
	
	oclMat rayCountChannel = oclMat(image.size(), channels[0].depth(), Scalar(1));
	channels.push_back(rayCountChannel);
	
	ocl::merge(channels, image);
}


void normalizeByRayCount(oclMat& image)
{
	vector<oclMat> channels;
	ocl::split(image, channels);

	int rayCountChannelIndex = channels.size() - 1;
	for (int i = 0; i < rayCountChannelIndex; i++)
	{
		ocl::divide(channels[i], channels[rayCountChannelIndex], channels[i]);
	}
	channels.pop_back();

	ocl::merge(channels, image);
}