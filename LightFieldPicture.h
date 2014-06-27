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
	static const int IMAGE_TYPE;

	LfpLoader loader;
	Mat rawImage;
	vector<Mat> subapertureImages;

	static Mat convertBayer2RGB(const Mat bayerImage);
	static Mat rectifyLensGrid(const Mat hexagonalLensGrid, LfpLoader metadata);
	Mat generateSubapertureImage(const unsigned short u, const unsigned short v);
public:
	Size SPARTIAL_RESOLUTION;
	Size ANGULAR_RESOLUTION;

	LightFieldPicture(void);
	LightFieldPicture(const string& pathToFile);
	~LightFieldPicture(void);

	Vec3f getLuminance(unsigned short x, unsigned short y, unsigned short u, unsigned short v);
	Vec3f getSubpixelLuminance(unsigned short x, unsigned short y, unsigned short u, unsigned short v);
	Vec3f getLuminanceF(float x, float y, float u, float v);
	Mat getSubapertureImage(const unsigned short u, const unsigned short v);
	Mat getSubapertureImageF(const double u, const double v);
	Mat getRawImage();

	double getRawFocalLength();
};

