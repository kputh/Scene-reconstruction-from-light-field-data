#pragma once

#include "RGBDMerger.h"

class RGBDMerger1 :
	public RGBDMerger
{
public:
	RGBDMerger1(void);
	~RGBDMerger1(void);

	Mat merge(vector<Mat> maps);
};

