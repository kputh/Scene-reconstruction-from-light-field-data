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
	static const double ZERO_THRESHOLD;
	static const Mat TEST_POINTS;
	static const Mat R90;

	FeatureDetector* detector;
	DescriptorExtractor* extractor;
	DescriptorMatcher* matcher;

public:
	CameraPoseEstimator1(void);
	~CameraPoseEstimator1(void);

	void estimateCameraPoses(const vector<Mat>& images,
		const Mat& calibrationMatrix);
};

