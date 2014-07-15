#pragma once

#include <opencv2/ocl/ocl.hpp>
#include "ImageRenderer.h"

/**
 * A refocus algorithm for rendering images from light fields. It is based on
 * equation 4.2 from Ren Ng's dissertation "Digital Light Field Photography".
 *
 * This algorithms works by shifting and adding the individual sub-aperture
 * images.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-06-20
 */
class ImageRenderer1 :
	public ImageRenderer
{
	static const float ACCUMULATOR_SCALE;
	static const Point2f UNIT_VECTORS[3];

	double weight;
	Size imageSize;
	int imageType;
	Vec2f angularCorrection;
	Vec2f fromCornerToCenter;
	Rect cutRect;

public:
	ImageRenderer1(void);
	~ImageRenderer1(void);

	void setLightfield(LightFieldPicture lightfield);
	void setAlpha(float alpha);

	oclMat renderImage() const;
};