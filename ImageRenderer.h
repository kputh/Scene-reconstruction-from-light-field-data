#pragma once

#include <opencv2/core/core.hpp>
#include "LightFieldFromLfpFile.h"

/**
 * The abstract base class for any algorithm which can render an image from a
 * light field.
 *
 * Rendering is based on the pinhole camera model. Not every algorithm uses
 * every camera parameter.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-06-20
 */
class ImageRenderer
{
protected:
	LightFieldFromLfpFile lightfield;
	double focalLength;
	Vec2i pinholePosition;
public:
	ImageRenderer(void);
	~ImageRenderer(void);

	LightFieldFromLfpFile getLightfield();
	void setLightfield(LightFieldFromLfpFile lightfield);
	double getFocalLength();
	void setFocalLength(double focalLength);
	Vec2i getPinholePosition();
	void setPinholePosition(Vec2i pinholePosition);

	virtual Mat getImage() =0;
};

