#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include "lightfield.h"

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
	Mat rawImage;

	static Mat convertBayer2RGB(const Mat bayerImage);
	static Mat rectifyLensGrid(const Mat hexagonalLensGrid);
public:
	static const Size SPARTIAL_RESOLUTION;
	static const Size ANGULAR_RESOLUTION;

	LightFieldFromLfpFile(void);
	LightFieldFromLfpFile(const string& pathToFile);
	~LightFieldFromLfpFile(void);

	Vec3s getLuminance(const unsigned short x, const unsigned short y, const unsigned short u, const unsigned short v);
	Mat getSubapertureImage(const unsigned short u, const unsigned short v);
	Mat getAllSubaperturesInOneImage();
	Mat getRawImage();
};

