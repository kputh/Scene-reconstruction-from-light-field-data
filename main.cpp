#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // ToDo remove
#include <iostream>

#include "LightFieldFromLfpFile.h"

using namespace cv;
using namespace std;

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

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat rawImage, subapertureImage, image1, image2, image4, image14;
	try {
		double t = (double)getTickCount();

		LightFieldFromLfpFile lf(argv[1]);

		rawImage = lf.getRawImage();
		image1 = lf.getImage(0.0068200001716613766 * 1.0);
		/*
		image14 = lf.getImage(0.0068200001716613766 * 0.25);
		image4 = lf.getImage(0.0068200001716613766 * 4.0);
		*/

		image1 = lf.getImage(0.0068200001716613766, -150, 0);
		image2 = lf.getImage(0.0068200001716613766, 150, 0);

		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Times passed in seconds: " << t << endl;
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

	cout << "Loading of file at " << argv[1] << " successful." << endl;

	// save image to file
	saveImageToPNGFile("leftImage.png", image1);
	saveImageToPNGFile("rightImage.png", image2);

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

	waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}