#pragma once
#include "cameraposeestimator.h"
class CameraPoseEstimator1 :
	public CameraPoseEstimator
{
	FeatureDetector* detector;
	DescriptorExtractor* extractor;
	//Feature2D* detectorAndExtractor;
	DescriptorMatcher* matcher;

public:
	CameraPoseEstimator1(void);
	~CameraPoseEstimator1(void);

	vector<Mat> estimateCameraPoses(const vector<Mat>& images,
		const Mat& calibrationMatrix) const;
};

