#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // ToDo remove
#include <iostream>

#include "LightFieldFromLfpFile.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
	try {
		LightFieldFromLfpFile lf(argv[1]);
		//image = lf.getSubapertureImage(5, 5);
		//image = lf.getAllSubaperturesInOneImage();
		image = lf.getRawImage();
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

	//
	/*rectify hexagonal tiles
	Mat src_gray;
	cvtColor( image, src_gray, CV_RGB2GRAY );
	src_gray.convertTo(src_gray, CV_8UC1, 1./256.);

	vector<Vec3f> circles;
	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 4, 6 );

	cout << circles.size() << " circles found." << endl;
	Mat circleImage = Mat::zeros(image.size(), image.type());
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( circleImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
	   circle( circleImage, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}
	*/

	cout << "Loading of file at " << argv[1] << " successful." << endl;
	cout << "Raw sensor image has " << image.size().width << " x " << image.size().height << " pixels." << endl;
	//cout << "Displaying sub-aperture image at angular coordinates (5, 5)." << endl;
	//cout << "Displaying raw image: de-bayered and rectified" << endl;

	// save image to file
	string fileName = "out.png";
	imwrite(fileName, image);
	cout << "Image saved as file." << endl;

	/*
	// show image
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display. (original size)
    //namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display. (scale down size)
    imshow( "Display window", image );                   // Show our image inside it.
	*/
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}