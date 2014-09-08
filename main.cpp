#define _USE_MATH_DEFINES	// for math constants in C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <cmath>
#include <string>
#include <iostream>

#include "Util.h"
#include "LightFieldPicture.h"
#include "ImageRenderer.h"
#include "ImageRenderer1.h"
#include "ImageRenderer4.h"
#include "DepthEstimator1.h"
#include "CDCDepthEstimator.h"
#include "DepthToPointTranslator.h"
#include "DepthToPointTranslator1.h"
#include "CameraPoseEstimator.h"
#include "CameraPoseEstimator1.h"
#include "RGBDMerger.h"
#include "RGBDMerger1.h"
#include "ReconstructionPipeline.h"

using namespace cv;
using namespace std;

const int lfpCount = 7;
const string lfpPaths[] = {
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue1.lfp",
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue2.lfp",
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue3.lfp",
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue4.lfp",
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue5.lfp",
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue6.lfp",
	"C:\\Users\\Kai\\Downloads\\lfpextraction\\statue\\statue7.lfp"
};

// store compiled ocl kernels in this path
const char KERNEL_PATH[] = "C:\\Users\\Kai\\Downloads\\opencv_ocl_kernels\\";

// renders a series of images from a LightFieldPicture and displays or saves them
void showRefocusSeries(const LightFieldPicture& lightfield)
{
	ImageRenderer* renderer = new ImageRenderer4();
	renderer->setLightfield(lightfield);

	Mat image;
	string windowName;
	for (int i = -20; i < lightfield.getLambdaInfinity() + 1.; i += 1)
	{
		renderer->setAlpha(i);
		renderer->renderImage().download(image);

		/*
		const string path = "C:\\Users\\Kai\\Downloads\\lfpextraction\\kranhaus";
		saveImageToPNGFile(path + to_string((long double)i) + ".png", image);
		*/
		/**/
		windowName = "refocused image with alpha = " + to_string((long double) i);
		namedWindow(windowName, WINDOW_AUTOSIZE);// Create a window for display. (scale down size)
		imshow(windowName, image);
		/**/
	}

	waitKey(0);
}

// reconstruct a scene from light-field data and render it
void renderReconstructionFromImageSeries(const string lfpPaths[])
{
	LightFieldPicture* lightfield;
	CDCDepthEstimator* estimator = new CDCDepthEstimator;
	RGBDMerger* merger = new RGBDMerger1();

	Mat calibrationMatrix;
	vector<Mat> depthMaps = vector<Mat>(lfpCount);
	vector<Mat> confidenceMaps = vector<Mat>(lfpCount);
	vector<Mat> aifImages = vector<Mat>(lfpCount);

	for (int i = 0; i < lfpCount; i++)
	{
		lightfield = new LightFieldPicture(lfpPaths[i]);
		estimator->estimateDepth(*lightfield);
		depthMaps[i] = estimator->getDepthMap();
		confidenceMaps[i] = estimator->getConfidenceMap();
		aifImages[i] = estimator->getExtendedDepthOfFieldImage();
	}

	calibrationMatrix = lightfield->getCalibrationMatrix();
	merger->merge(aifImages, depthMaps, confidenceMaps, calibrationMatrix);

	visualizePointCloud(merger->pointCloud, merger->pointColors);
}

// loads raw.lfp files using lfpPaths
vector<LightFieldPicture> loadLightFieldPictures()
{
	vector<LightFieldPicture> lfps = vector<LightFieldPicture>(lfpCount);
	for (int i = 0; i < lfpCount; i++)
	{
		lfps.at(i) = LightFieldPicture(lfpPaths[i]);
	}

	cout << lfpCount << " light-field files loaded" << endl;
	return lfps;
}

void testCameraPoseEstimation()
{
	vector<LightFieldPicture> lightfields = loadLightFieldPictures();

	vector<Mat> images = vector<Mat>(lightfields.size());
	ImageRenderer* renderer = new ImageRenderer4();
	renderer->setAlpha(1.1);
	for (int i = 0; i < images.size(); i++)
	{
		renderer->setLightfield(lightfields.at(i));
		renderer->renderImage().download(images.at(i));
	}

	Mat calibrationMatrix = lightfields.at(0).getCalibrationMatrix();
	CameraPoseEstimator* poseEstimator = new CameraPoseEstimator1();
	double t0 = (double)getTickCount();
	poseEstimator->estimateCameraPoses(images, calibrationMatrix);
	double t1 = (double)getTickCount();

	double d0 = (t1 - t0) / getTickFrequency();
	cout << "Camera pose estimation took " << d0 << " seconds." << endl;

	visualizeCameraTrajectory(*poseEstimator, Matx33d(calibrationMatrix));
}

void testPipeline()
{
	vector<LightFieldPicture> lightfields = loadLightFieldPictures();
	ReconstructionPipeline* pipeline = new ReconstructionPipeline();
	pipeline->reconstructScene(lightfields);

	visualizePointCloud(pipeline->pointCloud, pipeline->pointColors);
}

void testDepthEstimation(const LightFieldPicture& lightfield)
{
	CDCDepthEstimator* estimator = new CDCDepthEstimator();
	estimator->estimateDepth(lightfield);

	ImageRenderer* renderer = new ImageRenderer4();
	Mat image;
	renderer->setLightfield(lightfield);
	renderer->setAlpha(1.0);
	renderer->renderImage().download(image);

	/*
	saveImageToPNGFile("C:\\Users\\Kai\\Downloads\\lfpextraction\\naturalImage.png",
		image);
	saveImageToPNGFile("C:\\Users\\Kai\\Downloads\\lfpextraction\\depthMap.png",
		estimator->getDepthMap());
	saveImageToPNGFile("C:\\Users\\Kai\\Downloads\\lfpextraction\\confidenceMap.png",
		estimator->getConfidenceMap());
	saveImageToPNGFile("C:\\Users\\Kai\\Downloads\\lfpextraction\\allInFocusImage.png",
		estimator->getExtendedDepthOfFieldImage());
	*/

	Rect rect = Rect(5, 5, image.cols - 10, image.rows - 10);
	Mat depthMap = Mat(estimator->getDepthMap(), rect);
	Mat aifImage = Mat(estimator->getExtendedDepthOfFieldImage(), rect);

	DepthToPointTranslator* translator = new DepthToPointTranslator1();
	Mat pointCloud = translator->translateDepthToPoints(depthMap,
		lightfield.getCalibrationMatrix(),
		Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1));

	visualizePointCloud(pointCloud, aifImage);
}

int main( int argc, char** argv )
{
	ocl::setBinaryPath(KERNEL_PATH);

	if(argc != 2)
	{
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat rawImage, subapertureImage, image1, image2, image4, image14;
	oclMat ocl1, ocl2;
	try {
		//LightFieldPicture* lf = new LightFieldPicture("C:\\Users\\Kai\\Downloads\\lfpextraction\\fence.lfp");

		//showRefocusSeries(*lf);
		//testDepthEstimation(*lf);
		
		//testCameraPoseEstimation();
		testPipeline();

		/*
		LightFieldPicture* lf = new LightFieldPicture(argv[1]);
		showRefocusSeries(*lf);
		*/
		
		//renderReconstructionFromImageSeries(lfpPaths);
		/*
		double t0 = (double)getTickCount();
		LightFieldPicture lf(argv[1]);
		double t1 = (double)getTickCount();

		double d0 = (t1 - t0) / getTickFrequency();
		cout << "Loading of file at " << argv[1] << " successful." << endl;
		cout << "Loading of light field took " << d0 << " seconds." << endl;

		ImageRenderer1 renderer = ImageRenderer1();
		renderer.setAlpha(-0.5);
		renderer.setLightfield(*lf);

		//t0 = (double)getTickCount();
		ocl1 = renderer.renderImage();
		//t1 = (double)getTickCount();

		d0 = (t1 - t0) / getTickFrequency();
		cout << "Rendering took " << d0 << " seconds." << endl;

		ocl1.download(image1);
		saveImageToPNGFile("C:\\Users\\Kai\\Downloads\\lfpextraction\\Banding.png",
			image1);

		CDCDepthEstimator* estimator = new CDCDepthEstimator;

		t0 = (double)getTickCount();
		ocl2 = estimator->estimateDepth(lf);
		t1 = (double)getTickCount();

		d0 = (t1 - t0) / getTickFrequency();
		cout << "Depth estimation took " << d0 << " seconds." << endl;
		*/

		/*
		ocl2.download(image1);
		normalize(image1, image1, 0, 1, NORM_MINMAX);
		string window0 = "normalized depth";
		namedWindow(window0, WINDOW_AUTOSIZE);// Create a window for display. (scale down size)
		imshow(window0, image1);

		waitKey(0);
		*/

		/*
		ocl1.download(image1);
		saveImageToPNGFile("depthMap.png", image1);
		
		estimator->getConfidenceMap().download(image1);
		saveImageToPNGFile("confidenceMap.png", image1);

		estimator->getExtendedDepthOfFieldImage().download(image1);
		saveImageToPNGFile("all-in-focus-image.png", image1);
		*/	
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

	return 0;
}