#include <iostream>	// for debugging
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>	// for debugging
#include "ImageRenderer3.h"
#include "DepthEstimator1.h"


const float DepthEstimator1::FOCAL_LENGTH = 0.0068200001716613766;


DepthEstimator1::DepthEstimator1(void)
{
}


DepthEstimator1::~DepthEstimator1(void)
{
}


// from GitHub: Itseez/opencv/opencv/samples/cpp/simpleflow_demo.cpp
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}


Mat DepthEstimator1::estimateDepth(const LightFieldPicture lightfield)
{
	// render images
	ImageRenderer3 renderer = ImageRenderer3();
	renderer.setLightfield(lightfield);
	renderer.setFocalLength(DepthEstimator1::FOCAL_LENGTH);

	const Vec2i leftPosition = Vec2i(-5,0);
	renderer.setPinholePosition(leftPosition);
	Mat image1 = renderer.renderImage();
	cvtColor(image1, image1, CV_RGB2GRAY);
	image1.convertTo(image1, CV_8UC1, 255.0);
		
	const Vec2i rightPosition = Vec2i(5,0);
	renderer.setPinholePosition(rightPosition);
	Mat image2 = renderer.renderImage();
	cvtColor(image2, image2, CV_RGB2GRAY);
	image2.convertTo(image2, CV_8UC1, 255.0);

	// compute optical flow
	Mat opticalFlow;
	calcOpticalFlowFarneback(image1, image2, opticalFlow, 0.5, 3, 15, 3, 5, 1.2, 0);

	Mat flowMap;
	image1.copyTo(flowMap);
	drawOptFlowMap(opticalFlow, flowMap, 16, 1.5, Scalar(0, 255, 0));

	/* debug */
	const int windowFlags = WINDOW_NORMAL;
	const string window1 = "image1";
	namedWindow(window1, windowFlags);
	imshow(window1, image1);

	const string window2 = "image2";
	namedWindow(window2, windowFlags);
	imshow(window2, image2);
	
	const string window3 = "optical flow";
	namedWindow(window3, windowFlags);
	imshow(window3, flowMap);

	//cout << "optical flow = " << opticalFlow << endl;
	
	waitKey(0);
	/* end of debug code */

	return opticalFlow;
}