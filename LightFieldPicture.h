#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/ocl/ocl.hpp>
#include "lightfield.h"
#include "LfpLoader.h"

using namespace std;
using namespace cv;
using namespace ocl;

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
	vector<oclMat> subapertureImages;

	static Mat demosaicImage(const Mat& bayerImage);
	static Mat rectifyLensGrid(const Mat& hexagonalLensGrid,
		const LfpLoader& metadata);
	Mat generateSubapertureImage(const unsigned short u,
		const unsigned short v) const;
public:
	Size SPARTIAL_RESOLUTION;
	Size ANGULAR_RESOLUTION;

	LightFieldPicture(void);
	LightFieldPicture(const string& pathToFile);
	~LightFieldPicture(void);

	luminanceType getLuminance(unsigned short x, unsigned short y,
		unsigned short u, unsigned short v) const;
	luminanceType getSubpixelLuminance(unsigned short x, unsigned short y,
		unsigned short u, unsigned short v) const;
	luminanceType getLuminanceF(float x, float y, float u, float v) const;
	oclMat getSubapertureImageI(const unsigned short u, const unsigned short v) const;
	oclMat getSubapertureImageF(const double u, const double v) const;
	Mat getRawImage() const;

	double getRawFocalLength() const;
};

