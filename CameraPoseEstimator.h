#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

/**
 * The abstract base class for any algorithm which can estimate a series of camera
 * poses from a sequence of images.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-28
 */
class CameraPoseEstimator
{
	typedef Mat rotationType;
	typedef Vec4f translationType;

public:
	vector<rotationType> rotations;
	vector<translationType> translations;

	CameraPoseEstimator(void);
	~CameraPoseEstimator(void);

	virtual void estimateCameraPoses(const vector<Mat>& images,
		const Mat& calibrationMatrix) const =0;
};

