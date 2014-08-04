#include <opencv2\features2d\features2d.hpp>
#include "CameraPoseEstimator.h"
#include "CameraPoseEstimator1.h"
#include "DepthToPointTranslator.h"
#include "DepthToPointTranslator1.h"
#include "RGBDMerger1.h"


RGBDMerger1::RGBDMerger1(void)
{
	this->poseEstimator = new CameraPoseEstimator1();
	this->d2pTranslator = new DepthToPointTranslator1();
}


RGBDMerger1::~RGBDMerger1(void)
{
	delete this->poseEstimator;
	delete this->d2pTranslator;
}


Mat RGBDMerger1::merge(const vector<Mat>& images, const vector<Mat>& maps,
	const Mat& calibrationMatrix)
{
	// 1) estimate camera poses
	poseEstimator->estimateCameraPoses(images, calibrationMatrix);

	// 2) triangulate points
	vector<Mat> partialReconstructions = vector<Mat>();

	for (int i = 0; i < maps.size(); i++)
	{
		partialReconstructions.push_back(
			d2pTranslator->translateDepthToPoints(maps.at(i), calibrationMatrix,
				poseEstimator->rotations.at(i), poseEstimator->translations.at(i))
		);
	}

	// 3) merge and remove redundancies
	// TODO ...
	Mat completeReconstruction = partialReconstructions.at(0).clone();
	Mat luminanceMap = images.at(0).clone();
	for (int i = 1; i < partialReconstructions.size(); i++)
	{
		completeReconstruction.push_back(partialReconstructions.at(i));

		luminanceMap.push_back(images.at(i));
	}

	// or something totally different

	/*
	FeatureDetector* detector = new DenseFeatureDetector();
	DescriptorExtractor* extractor = new ORB();

	vector<vector<KeyPoint>> keyPoints;
	vector<Mat> descriptors;

	// 1) "detect" dense features
	detector->detect(maps, keyPoints);
	// 2) extract descriptors
	extractor->compute(maps, keyPoints, descriptors);
	// 3) match descriptors
	// TODO
	DescriptorMatcher* matcher;
	*/

	this->pointCloud = completeReconstruction;
	this->pointColors = luminanceMap;

	return completeReconstruction;
}