#pragma once

#include "DepthEstimator.h"

/**
 * Unfinished implementation using ImageRenderer3 to render a stereo image pair
 * and cv::StereoBM (block matching) to compute a disparity map.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-09-08
 */
class StereoBMDisparityEstimator :
	public DepthEstimator
{
	static const float ALPHA;
	static const Vec2i LEFT_POSITION;
	static const Vec2i RIGHT_POSITION;

public:
	StereoBMDisparityEstimator(void);
	~StereoBMDisparityEstimator(void);

	Mat estimateDepth(const LightFieldPicture lightfield);
};

