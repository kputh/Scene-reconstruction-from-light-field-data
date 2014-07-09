#pragma once

#include "ImageRenderer.h"

/**
 * A perspective-shift and refocus algorithm for rendering images from light
 * fields. It is derived from equation 5 from "Light field photography with a
 * hand-held plenoptic camera" by Ng et al. (2005).
 *
 * This algorithms works by shifting, weighing and adding the individual
 * sub-aperture images. The aperture function is still present in the equation,
 * meaning each sub-aperture image is multiplied by a weight before adding all
 * sub-aperture images up. A bivariate normal distribution was chosen as
 * aperture function. The normal distribution is centered on the sub-aperture.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-06-20
 */
class ImageRenderer3 :
	public ImageRenderer
{
public:
	ImageRenderer3(void);
	~ImageRenderer3(void);

	oclMat renderImage() const;
};