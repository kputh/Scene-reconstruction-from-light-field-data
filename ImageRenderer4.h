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
class ImageRenderer4 :
	public ImageRenderer
{
	double weight;

public:
	ImageRenderer4(void);
	~ImageRenderer4(void);

	void setLightfield(const LightFieldPicture& lightfield);
	void setAlpha(float alpha);

	oclMat renderImage() const;
};