#pragma once

#include <opencv2/ocl/ocl.hpp>
#include "LightFieldPicture.h"

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
	LightFieldPicture lightfield;
	float alpha;
	Vec2i pinholePosition;
public:
	ImageRenderer(void);
	~ImageRenderer(void);

	LightFieldPicture getLightfield() const;
	void setLightfield(LightFieldPicture lightfield);
	float getAlpha() const;
	void setAlpha(float alpha);
	Vec2i getPinholePosition() const;
	void setPinholePosition(Vec2i pinholePosition);

	virtual oclMat renderImage() const =0;
};

