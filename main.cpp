#define _USE_MATH_DEFINES	// for math constants in C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // ToDo remove
#include <cmath>
#include <string>
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

void saveImageArc(LightFieldFromLfpFile lightfield, string sourceFileName, int imageCount)
{
	float angle, x, y;
	float radius = 100;
	float f = 0.0068200001716613766;

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
		image = lightfield.getImage2(f, x, y);
		imageFileName = sourceFileName + to_string((long double)i) + fileExtension;
		saveImageToPNGFile(imageFileName, image);
	}
	cout << "image arc saved" << endl;
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
		cout << "Loading of file at " << argv[1] << " successful." << endl;
	
		saveImageArc(lf, string(argv[1]), 8);

		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Times passed in seconds: " << t << endl;
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

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