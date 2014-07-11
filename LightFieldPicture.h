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
	static const Point IMAGE_ORIGIN;

	LfpLoader loader;
	Mat rawImage, processesImage;
	vector<oclMat> subapertureImages;
	oclMat subapertureImageAtlas;

	static Mat demosaicImage(const Mat& bayerImage);
	static void rectifyLensGrid(Mat& hexagonalLensGrid,
		const LfpLoader& metadata);
	oclMat generateSubapertureImage(const unsigned short u,
		const unsigned short v) const;
	static oclMat extractSubapertureImageAtlas(const Mat& hexagonalLensGrid,
		const LfpLoader& metadata);

	Rect validSpartialCoordinates;
	double microLensRadiusInPixels;
	Vec2f fromLensCenterToOrigin;

public:
	typedef float luminanceType;
	static const int IMAGE_TYPE;

	static const luminanceType ZERO_LUMINANCE;

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
	oclMat getSubapertureImageAtlas() const;

	double getRawFocalLength() const;
};

