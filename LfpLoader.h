#pragma once

#include <opencv2/core/core.hpp>

using namespace cv;

class LfpLoader
{
	static const char IMAGE_KEY[];
	static const char WIDTH_KEY[];
	static const char HEIGHT_KEY[];
public:
	static Mat load(const string& path);
};