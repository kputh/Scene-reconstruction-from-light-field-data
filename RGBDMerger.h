#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

/**
 * The abstract base class for any algorithm which can merge several RGB+D maps
 * into a single point cloud.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-03
 */
class RGBDMerger
{
public:
	RGBDMerger(void);
	~RGBDMerger(void);

	virtual Mat merge(vector<Mat> maps) =0;
};

