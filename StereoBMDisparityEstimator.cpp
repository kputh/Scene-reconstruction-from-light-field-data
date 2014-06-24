#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>	// for debugging
#include "StereoBMDisparityEstimator.h"
#include "ImageRenderer3.h"


StereoBMDisparityEstimator::StereoBMDisparityEstimator(void)
{
}


StereoBMDisparityEstimator::~StereoBMDisparityEstimator(void)
{
}


Mat StereoBMDisparityEstimator::estimateDepth(const LightFieldFromLfpFile lightfield)
{
	ImageRenderer3 renderer = ImageRenderer3();
	renderer.setLightfield(lightfield);
	renderer.setFocalLength(0.0068200001716613766);

	const Vec2i leftPosition = Vec2i(0,0);
	renderer.setPinholePosition(leftPosition);
	Mat image1 = renderer.renderImage();
	cvtColor(image1, image1, CV_RGB2GRAY);
	image1.convertTo(image1, CV_8UC1, 255.0);
		
	const Vec2i rightPosition = Vec2i(5,0);
	renderer.setPinholePosition(rightPosition);
	Mat image2 = renderer.renderImage();
	cvtColor(image2, image2, CV_RGB2GRAY);
	image2.convertTo(image2, CV_8UC1, 255.0);

	Mat disparity;
	StereoBM stereo = StereoBM(StereoBM::BASIC_PRESET, 32, 21);
	stereo(image1, image2, disparity, CV_32F);

	/* debug */
	const int windowFlags = WINDOW_NORMAL;
	const string window1 = "image1";
	namedWindow(window1, windowFlags);
	imshow(window1, image1);

	const string window2 = "image2";
	namedWindow(window2, windowFlags);
	imshow(window2, image2);
		
	const string window3 = "disparity map";
	namedWindow(window3, windowFlags);
	imshow(window3, disparity);

	waitKey(0);
	/* end of debug code */

	return disparity;
}
