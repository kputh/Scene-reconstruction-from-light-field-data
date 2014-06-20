#pragma once

#include "ImageRenderer.h"

/**
 * A perspective-shift and refocus algorithm for rendering images from light
 * fields. It is based on equation 7 from "Light field photography with a
 * hand-held plenoptic camera" by Ng et al. (2005).
 *
 * This algorithm works by selecting a single pixel from each microlens' image.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-06-20
 */
class ImageRenderer2 :
	public ImageRenderer
{
public:
	ImageRenderer2(void);
	~ImageRenderer2(void);

	Mat getImage();
};

