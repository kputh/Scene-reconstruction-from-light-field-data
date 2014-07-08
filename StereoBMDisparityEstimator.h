#pragma once

#include "DepthEstimator.h"

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

