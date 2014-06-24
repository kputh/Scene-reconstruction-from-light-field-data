#define _USE_MATH_DEFINES	// for math constants in C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // ToDo remove
#include <cmath>
#include <string>
#include <iostream>

#include "Util.h"
#include "LightFieldFromLfpFile.h"
#include "ImageRenderer.h"
#include "ImageRenderer1.h"
#include "ImageRenderer2.h"
#include "ImageRenderer3.h"

using namespace cv;
using namespace std;

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

		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Times passed in seconds: " << t << endl;
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

	//saveImageToPNGFile(string(argv[1]) + string(".png"), image1);
	cin.ignore();
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