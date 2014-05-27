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
	virtual cv::Scalar getLuminance(const unsigned short x, const unsigned short y, const unsigned short u, const unsigned short v) = 0;
	virtual cv::Mat getSubapertureImage(const double u, const double v) = 0;
	virtual cv::Mat getEpipolarImage(const double one, const double other) = 0;
	virtual cv::Mat getImage(const double focalLength) = 0;
	virtual cv::Mat getAllInFocusImage() = 0;
};

