#define _USE_MATH_DEFINES	// for math constants in C++

#include <cmath>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\viz\vizcore.hpp>

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


oclMat extractRayCountMat(const oclMat& image)
{
	oclMat img = image;
	if (image.channels() == 3)
		cvtColor(image, img, CV_RGB2GRAY);

	oclMat rayCountMat;
	ocl::threshold(img, rayCountMat, 0, 1, THRESH_BINARY);

	return rayCountMat;
}


void normalizeByRayCount(oclMat& image, const oclMat& rayCountMat)
{
	vector<oclMat> channels;
	ocl::split(image, channels);

	for (int i = 0; i < channels.size(); i++)
	{
		ocl::divide(channels[i], rayCountMat, channels[i]);
	}
	
	ocl::merge(channels, image);
}


void normalize(oclMat& mat)
{
	vector<oclMat> channels;
	ocl::split(mat, channels);

	double minVal, maxVal, totalMaxVal = -1;
	for(int i = 0; i < channels.size(); i++)
	{
		minMaxLoc(channels.at(i), &minVal, &maxVal);
		totalMaxVal = max(totalMaxVal, maxVal);
	}

	ocl::multiply(1. / totalMaxVal, mat, mat);
}

void visualizeCameraTrajectory(const CameraPoseEstimator& estimator,
	const Matx33d& calibrationMatrix)
{
	/// Create a window
	viz::Viz3d myWindow("Coordinate Frame");

	/// Add coordinate axes
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	vector<Affine3<double>> cameraPath = vector<Affine3<double>>();
	Mat translation, rotation;
	double x, y, z;
	Vec3d position, vDirection;
	const Vec3d up = Vec3d(0, 1, 0);
	Mat mDirection, unity3 = (Mat_<double>(3, 1) << 0, 0, 1);
	for (int i = 0; i < estimator.rotations.size(); i++)
	{
		translation = estimator.translations.at(i);
		x = translation.at<double>(0, 0);
		y = translation.at<double>(1, 0);
		z = translation.at<double>(2, 0);
		position = Vec3d(x,y,z);

		mDirection = estimator.rotations.at(i) * unity3;
		x = mDirection.at<double>(0,0);
		y = mDirection.at<double>(1,0);
		z = mDirection.at<double>(2,0);
		vDirection = Vec3d(x, y, z);

		cameraPath.push_back(
			viz::makeCameraPose(position, vDirection, up));
	}
	/*
	viz::WTrajectory trajectory = viz::WTrajectory(cameraPath,
		viz::WTrajectory::BOTH);
		*/
	//viz::WTrajectorySpheres trajectory = viz::WTrajectorySpheres(cameraPath);
	viz::WTrajectoryFrustums trajectory = viz::WTrajectoryFrustums(cameraPath,
		calibrationMatrix);

	myWindow.showWidget("Camera trajectory", trajectory);
	myWindow.spin();
}

void visualizePointCloud(const Mat& pointCloud, const Mat& image)
{
	/// Create a window
	viz::Viz3d myWindow("Coordinate Frame");

	/// Add coordinate axes
	//myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	Mat colors;
	image.convertTo(colors, CV_8U, 255);
	viz::WCloud cloudWidget = viz::WCloud(pointCloud, colors);

	const int width		= pointCloud.size().width;
	const int height	= pointCloud.size().height;
	vector<int> polygons = vector<int>();
	for (int y = 0; y < height - 1; y++)
	for (int x = 0; x < width - 1; x++)
	{
		polygons.push_back(3);
		polygons.push_back(y * width + x);
		polygons.push_back(y * width + (x + 1));
		polygons.push_back((y + 1) * width + x);

		polygons.push_back(3);
		polygons.push_back((y + 1) * width + (x + 1));
		polygons.push_back((y + 1) * width + x);
		polygons.push_back(y * width + (x + 1));
	}

	//Mat p = Mat(polygons, false);
	viz::WMesh meshWidget = viz::WMesh(pointCloud.reshape(0, 1), polygons,
		colors.reshape(0, 1));

	myWindow.showWidget("Point cloud", cloudWidget);
	//myWindow.showWidget("Mesh", meshWidget);
	
	myWindow.spin();
}