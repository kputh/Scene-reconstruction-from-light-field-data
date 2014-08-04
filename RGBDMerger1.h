#pragma once

#include "RGBDMerger.h"

class RGBDMerger1 :
	public RGBDMerger
{
	CameraPoseEstimator* poseEstimator;
	DepthToPointTranslator* d2pTranslator;

public:
	RGBDMerger1(void);
	~RGBDMerger1(void);

	Mat merge(const vector<Mat>& images, const vector<Mat>& maps,
		const Mat& calibrationMatrix);
};

