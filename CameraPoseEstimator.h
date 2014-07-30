#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

class CameraPoseEstimator
{
public:
	CameraPoseEstimator(void);
	~CameraPoseEstimator(void);

	virtual vector<Mat> estimateCameraPoses(const vector<Mat>& images,
		const Mat& calibrationMatrix) const =0;
};

