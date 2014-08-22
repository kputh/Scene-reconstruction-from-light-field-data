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
	this->detector = new DenseFeatureDetector();
	this->extractor = new ORB();
	//this->matcher = new ocl::BruteForceMatcher_OCL_base();
	this->matcher = new BFMatcher(NORM_HAMMING, true);
}


RGBDMerger1::~RGBDMerger1(void)
{
	delete this->poseEstimator;
	delete this->d2pTranslator;
	delete this->detector;
	delete this->extractor;
	delete this->matcher;
}


Mat RGBDMerger1::merge(const vector<Mat>& images, const vector<Mat>& depthMaps,
	const vector<Mat>& confidenceMaps, const Mat& calibrationMatrix)
{
	assert(images.size() > 1);

	// 1) estimate camera poses
	poseEstimator->estimateCameraPoses(images, calibrationMatrix);

	// 2) triangulate points
	vector<Mat> partialReconstructions = vector<Mat>(depthMaps.size());
	for (int i = 0; i < depthMaps.size(); i++)
	{
		partialReconstructions.at(i) = d2pTranslator->translateDepthToPoints(
			depthMaps.at(i), calibrationMatrix, poseEstimator->rotations.at(i),
			poseEstimator->translations.at(i));
	}

	// 3) find redundancies
	vector<vector<KeyPoint>> keyPoints;
	vector<Mat> descriptors;
	vector<DMatch> matches;
	vector<Mat> masks = vector<Mat>(images.size());
	for (int i = 0; i < masks.size(); i++)
		masks[i] = Mat(images.at(i).size(), CV_8SC1, Scalar::all(255));

	detector->detect(images, keyPoints);
	extractor->compute(images, keyPoints, descriptors);

	Mat queryDescriptors;
	int i, j, matchIndex;
	DMatch match;
	Point2f pt1, pt2;
	float confidence1, confidence2;
	for (i = images.size() - 1; i >= 0; i--)
	{
		queryDescriptors = descriptors.at(i);
		descriptors.pop_back();

		matcher->clear();
		matcher->add(descriptors);

		matcher->match(queryDescriptors, matches);

		for (matchIndex = 0; matchIndex < matches.size(); matchIndex++)
		{
			match = matches.at(matchIndex);
			j = match.imgIdx;

			pt1 = keyPoints.at(i).at(match.queryIdx).pt;
			pt2 = keyPoints.at(j).at(match.trainIdx).pt;

			confidence1 = confidenceMaps.at(i).at<float>(pt1);
			confidence2 = confidenceMaps.at(j).at<float>(pt2);

			if (confidence1 < confidence2)
				masks.at(i).at<uchar>(pt1) = 0;
			else
				masks.at(j).at<uchar>(pt2) = 0;
		}
	}


	// 4) merge partial reconstructions
	Mat completeReconstruction = Mat();//partialReconstructions.at(0).clone();
	Mat luminanceMap = Mat();//images.at(0).clone();
	for (int i = 0; i < partialReconstructions.size(); i++)
	{
		// handle redundant points
		partialReconstructions.at(i).setTo(Scalar::all(0), masks.at(i) - 255);

		// join partial models
		completeReconstruction.push_back(partialReconstructions.at(i));
		luminanceMap.push_back(images.at(i));
	}

	this->pointCloud = completeReconstruction;
	this->pointColors = luminanceMap;

	return completeReconstruction;
}