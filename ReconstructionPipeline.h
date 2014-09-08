#pragma once

#include <vector>
#include <opencv2\core\core.hpp>
#include "LightFieldPicture.h"
#include "CDCDepthEstimator.h"
#include "RGBDMerger.h"

/**
 * The central class for the reconstruction pipeline.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-08-15
 */
class ReconstructionPipeline
{
	CDCDepthEstimator* estimator;
	RGBDMerger* merger;

public:
	Mat pointCloud, pointColors;

	ReconstructionPipeline(void);
	~ReconstructionPipeline(void);

	void reconstructScene(const vector<LightFieldPicture>& lightfields);
};

