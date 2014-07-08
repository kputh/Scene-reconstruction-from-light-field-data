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
public:
	ImageRenderer1(void);
	~ImageRenderer1(void);

	oclMat renderImageI();
	oclMat renderImageOCL();
};

