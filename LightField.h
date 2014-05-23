#pragma once

#include <opencv2/core/core.hpp>

/**
 * The data structure and abstract base class representing a light field.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-05-23
 */

class LightField
{
public:
	LightField(void);
	virtual ~LightField(void);
	virtual unsigned short getLuminance(double x, double y, double u, double v) = 0;
	virtual cv::Mat getSubapertureImage(double u, double v) = 0;
	virtual cv::Mat getEpipolarImage(double one, double other) = 0;
	virtual cv::Mat getImage(double focalLength) = 0;
	virtual cv::Mat getAllInFocusImage() = 0;
};

