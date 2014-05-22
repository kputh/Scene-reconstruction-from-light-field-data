#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "LfpLoader.h"

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
		image = LfpLoader::loadAsRGB(argv[1]);   // Read the file
	} catch (std::exception* e) {
		cerr << e->what() << endl;
		return -1;
	}

	cout << "Loading of file at " << argv[1] << " successful." << endl;
	cout << "Raw sensor image has " << image.size().width << " x " << image.size().height << " pixels." << endl;

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display. (original size)
    //namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display. (scale down size)
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}