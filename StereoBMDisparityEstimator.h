#pragma once

#include "DepthEstimator.h"

class StereoBMDisparityEstimator :
	public DepthEstimator
{
public:
	StereoBMDisparityEstimator(void);
	~StereoBMDisparityEstimator(void);

	Mat estimateDepth(const LightFieldFromLfpFile lightfield);
};

