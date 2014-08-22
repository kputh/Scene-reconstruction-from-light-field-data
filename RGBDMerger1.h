#pragma once

#include <opencv2\ocl\ocl.hpp>
#include <opencv2\features2d\features2d.hpp>
#include "RGBDMerger.h"
#include "CameraPoseEstimator.h"
#include "DepthToPointTranslator.h"

class RGBDMerger1 :
	public RGBDMerger
{
	CameraPoseEstimator* poseEstimator;
	DepthToPointTranslator* d2pTranslator;
	FeatureDetector* detector;
	DescriptorExtractor* extractor;
	//ocl::BruteForceMatcher_OCL_base* matcher;
	BFMatcher* matcher;

public:
	RGBDMerger1(void);
	~RGBDMerger1(void);

	Mat merge(const vector<Mat>& images, const vector<Mat>& depthMaps,
		const vector<Mat>& confidenceMaps, const Mat& calibrationMatrix);
};

