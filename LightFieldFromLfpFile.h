#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include "lightfield.h"
#include "LfpLoader.h"

using namespace std;
using namespace cv;

/**
 * The data structure and abstract base class representing a light field.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-05-27
 */
class LightFieldFromLfpFile /*:
	public LightField*/
{
	LfpLoader loader;
	Mat rawImage;

	static Mat convertBayer2RGB(const Mat bayerImage);
	static Mat rectifyLensGrid(const Mat hexagonalLensGrid, LfpLoader metadata);
	static Mat LightFieldFromLfpFile::adjustLuminanceSpace(const Mat image);
public:
	Size SPARTIAL_RESOLUTION;
	Size ANGULAR_RESOLUTION;

	LightFieldFromLfpFile(void);
	LightFieldFromLfpFile(const string& pathToFile);
	~LightFieldFromLfpFile(void);

	Vec3f getLuminance(const unsigned short x, const unsigned short y, const unsigned short u, const unsigned short v);
	Mat getSubapertureImage(const unsigned short u, const unsigned short v);
	Mat getImage(const double focalLength);
	Mat getImage(const double focalLength, const short x0, short y0);
	Mat getRawImage();
};

