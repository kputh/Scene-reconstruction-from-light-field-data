#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include "lightfield.h"
#include "LfpLoader.h"

using namespace std;
using namespace cv;

/**
 * The data structure for a light field from a Light Field Picture (*.lfp) file.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-05-27
 */
class LightFieldPicture /*:
	public LightField*/
{
	typedef Vec3f luminanceType;
	static const int IMAGE_TYPE;

	static const luminanceType ZERO_LUMINANCE;

	LfpLoader loader;
	Mat rawImage;
	vector<Mat> subapertureImages;

	static Mat demosaicImage(const Mat bayerImage);
	static Mat rectifyLensGrid(const Mat hexagonalLensGrid, LfpLoader metadata);
	Mat generateSubapertureImage(const unsigned short u, const unsigned short v);
public:
	Size SPARTIAL_RESOLUTION;
	Size ANGULAR_RESOLUTION;

	LightFieldPicture(void);
	LightFieldPicture(const string& pathToFile);
	~LightFieldPicture(void);

	luminanceType getLuminance(unsigned short x, unsigned short y,
		unsigned short u, unsigned short v);
	luminanceType getSubpixelLuminance(unsigned short x, unsigned short y,
		unsigned short u, unsigned short v);
	luminanceType getLuminanceF(float x, float y, float u, float v);
	Mat getSubapertureImageI(const unsigned short u, const unsigned short v);
	Mat getSubapertureImageF(const double u, const double v);
	Mat getRawImage();

	double getRawFocalLength();
};

