#pragma once

#include "CameraPoseEstimator.h"

/**
 * An implementation of camera pose estimation.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-28
 */
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

	void estimateCameraPoses(const vector<Mat>& images,
		const Mat& calibrationMatrix) const;
};

