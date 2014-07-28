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
#include "ImageRenderer2.h"
#include "ImageRenderer3.h"
#include "StereoBMDisparityEstimator.h"
#include "DepthEstimator1.h"
#include "CDCDepthEstimator.h"
#include "DepthToPointTranslator.h"
#include "DepthToPointTranslator1.h"

using namespace cv;
using namespace std;

const char KERNEL_PATH[] = "C:\\Users\\Kai\\Downloads\\opencv_ocl_kernels\\";

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
		double t0 = (double)getTickCount();
		LightFieldPicture lf(argv[1]);
		double t1 = (double)getTickCount();

		double d0 = (t1 - t0) / getTickFrequency();
		cout << "Loading of file at " << argv[1] << " successful." << endl;
		cout << "Loading of light field took " << d0 << " seconds." << endl;
		/*
		ImageRenderer1 renderer = ImageRenderer1();
		renderer.setAlpha(1.5);
		renderer.setLightfield(lf);

		t0 = (double)getTickCount();
		ocl1 = renderer.renderImage();
		t1 = (double)getTickCount();

		d0 = (t1 - t0) / getTickFrequency();
		cout << "Rendering took " << d0 << " seconds." << endl;
		
		ocl1.download(image1);
		//lf.getSubapertureImageI(5,5).download(image1);
		string window1 = "refocused image";
		namedWindow(window1, WINDOW_NORMAL);// Create a window for display. (scale down size)
		imshow(window1, image1);
		waitKey(0);
		*/
		
		CDCDepthEstimator* estimator = new CDCDepthEstimator;

		t0 = (double)getTickCount();
		ocl1 = estimator->estimateDepth(lf);
		t1 = (double)getTickCount();

		d0 = (t1 - t0) / getTickFrequency();
		cout << "Depth estimation took " << d0 << " seconds." << endl;

		DepthToPointTranslator* translator = new DepthToPointTranslator1();
		oclMat oclPoints = translator->translateDepthToPoints(ocl1, lf);

		/// Create a window
		viz::Viz3d myWindow("Coordinate Frame");

		/// Add coordinate axes
		//myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

		Mat points, image;
		estimator->getExtendedDepthOfFieldImage().download(image);
		image.convertTo(image, CV_8U, 255);
		oclPoints.download(points);
		viz::WCloud cloudWidget = viz::WCloud(points, image);

		const int width		= points.size().width;
		const int height	= points.size().height;
		vector<int> polygons = vector<int>();
		for (int y = 0; y < height - 1; y++)
		{
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
		}
		Mat p = Mat(polygons, false);
		viz::WMesh meshWidget = viz::WMesh(points.reshape(0, 1), polygons,
			image.reshape(0, 1));

		//myWindow.showWidget("cloud", cloudWidget);
		myWindow.showWidget("mesh", meshWidget);

		myWindow.spin();

		/*
		Mat m; ocl1.download(m); normalize(m, m, 0, 1, NORM_MINMAX);
		string window1 = "depth map";
		namedWindow(window1, WINDOW_NORMAL);// Create a window for display. (scale down size)
		imshow(window1, m);

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
			/*
		ImageRenderer3 renderer = ImageRenderer3();
		renderer.setLightfield(lf);
		renderer.setFocalLength(0.0068200001716613766);

		renderer.setPinholePosition(Vec2i(0,0));
		image1 = renderer.renderImage();
		cvtColor(image1, image1, CV_RGB2GRAY);
		image1.convertTo(image1, CV_8UC1, 255.0);
		
		renderer.setPinholePosition(Vec2i(5,0));
		image2 = renderer.renderImage();
		cvtColor(image2, image2, CV_RGB2GRAY);
		image2.convertTo(image2, CV_8UC1, 255.0);

		Mat disparity;
		StereoBM stereo = StereoBM(StereoBM::BASIC_PRESET, 16, 15);
		stereo(image1, image2, disparity, CV_32F);
		
		string window1 = "optical flow";
		namedWindow(window1, WINDOW_NORMAL);// Create a window for display. (scale down size)
		imshow(window1, image1);
		
		string window2 = "image2";

		namedWindow(window2, WINDOW_NORMAL);// Create a window for display. (scale down size)
		imshow(window2, image2);
		
		string window3 = "disparity map";
		namedWindow(window3, WINDOW_NORMAL);// Create a window for display. (scale down size)
		imshow(window3, disparity);

		waitKey(0);                                          // Wait for a keystroke in the window
		*/
		//saveImageArc(lf, string(argv[1]), 8);
		
		/*
		ImageRenderer1 renderer = ImageRenderer1();
		renderer.setLightfield(lf);
		renderer.setFocalLength(0.0068200001716613766);
		image1 = renderer.renderImage();
		*/
		/*
		string window1 = "difference image";
		namedWindow(window1, WINDOW_NORMAL);// Create a window for display. (scale down size)
		for (int u = 0; u < lf.ANGULAR_RESOLUTION.width; u++)
		{
			int v = 0;//for (int v = 0; v < lf.ANGULAR_RESOLUTION.height; v++)
			{
				cout << "(u, v) = (" << u << ", " << v << ")" << endl;
				image1 = lf.getSubapertureImage(u, v);
				//cout << "sub-aperture image = " << image1 << endl;
				imshow(window1, image1);                   // Show our image inside it.
				waitKey(0);                                          // Wait for a keystroke in the window
			}
		}
		*/
	
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

	//saveImageToPNGFile(string(argv[1]) + string(".png"), image1);
	//cin.ignore();
	// show image
	//namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display. (original size)
	
	/*
	string window1 = "difference image";
	namedWindow(window1, WINDOW_NORMAL);// Create a window for display. (scale down size)
	imshow(window1, image1);                   // Show our image inside it.
	*/
	/*
	namedWindow( "1.0x", WINDOW_NORMAL );// Create a window for display. (original size)
	imshow( "1.0x", image1 );                   // Show our image inside it.
	*/

	//waitKey(0);                                          // Wait for a keystroke in the window

	return 0;
}