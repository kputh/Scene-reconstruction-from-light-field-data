#include <iostream>					// for debugging messages
#include "ReconstructionPipeline.h"
#include "RGBDMerger1.h"


ReconstructionPipeline::ReconstructionPipeline(void)
{
	this->estimator = new CDCDepthEstimator;
	this->merger = new RGBDMerger1();
}


ReconstructionPipeline::~ReconstructionPipeline(void)
{
	delete this->estimator;
	delete this->merger;
}


void ReconstructionPipeline::reconstructScene(const vector<LightFieldPicture>&
	lightfields)
{
	int lfpCount = lightfields.size();
	vector<Mat> depthMaps = vector<Mat>(lfpCount);
	vector<Mat> confidenceMaps = vector<Mat>(lfpCount);
	vector<Mat> aifImages = vector<Mat>(lfpCount);

	//cout << "starting depth estimation" << endl;

	// estimate depth,...
	for (int i = 0; i < lfpCount; i++)
	{
		estimator->estimateDepth(lightfields.at(i));
		depthMaps[i] = estimator->getDepthMap();
		confidenceMaps[i] = estimator->getConfidenceMap();
		aifImages[i] = estimator->getExtendedDepthOfFieldImage();	// implicite download?
	}

	//cout << "starting model fusion" << endl;

	// merge partial reconstructions
	const Mat calibrationMatrix = lightfields.at(0).getCalibrationMatrix();
	merger->merge(aifImages, depthMaps, confidenceMaps, calibrationMatrix);

	// get results from merger
	this->pointCloud	= merger->pointCloud;
	this->pointColors	= merger->pointColors;
}

