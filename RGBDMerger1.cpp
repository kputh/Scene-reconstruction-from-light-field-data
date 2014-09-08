#include <iostream>		// for console output
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
	this->detector = new DenseFeatureDetector(1.f, 1, 0.1f, 1);
	this->extractor = new ORB();
	//this->matcher = new BFMatcher(NORM_HAMMING, true);
	this->matcher = new ocl::BruteForceMatcher_OCL_base(
		ocl::BruteForceMatcher_OCL_base::HammingDist);	// no cross-checking
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

	cout << "RGBDMerger::merge(): 1) estimate camera poses" << endl;
	// 1) estimate camera poses
	poseEstimator->estimateCameraPoses(images, calibrationMatrix);

	cout << "RGBDMerger::merge(): 2) triangulate points" << endl;
	// 2) triangulate points
	vector<Mat> partialReconstructions = vector<Mat>(depthMaps.size());
	for (int i = 0; i < depthMaps.size(); i++)
	{
		partialReconstructions.at(i) = d2pTranslator->translateDepthToPoints(
			depthMaps.at(i), calibrationMatrix, poseEstimator->rotations.at(i),
			poseEstimator->translations.at(i));
	}

	cout << "RGBDMerger::merge(): 3) find redundancies" << endl;
	// 3) find redundancies
	vector<vector<KeyPoint>> keyPoints;
	vector<Mat> descriptors;
	vector<DMatch> matches;
	vector<vector<DMatch>> matches12, matches21;

	Mat mask = Mat(images.at(0).size(), CV_8SC1, Scalar::all(0));
	rectangle(mask, Point(32, 32), Point(images.at(0).size().width - 31,
		images.at(0).size().height - 31), Scalar::all(255), CV_FILLED);
	vector<Mat> masks = vector<Mat>(images.size());
	for (int i = 0; i < masks.size(); i++)
		mask.copyTo(masks[i]);

	// prepare images for feature detection and descriptor extraction
	vector<Mat> bwImages = vector<Mat>(images.size());
	for (int i = 0; i < images.size(); i++)
	{
		cvtColor(images.at(i), bwImages.at(i), CV_RGB2GRAY);
		bwImages.at(i).convertTo(bwImages.at(i), CV_8UC1, 255);
	}

	detector->detect(bwImages, keyPoints);
	extractor->compute(bwImages, keyPoints, descriptors);
	
	// upload descriptors to GPU
	vector<oclMat> oclDescriptors = vector<oclMat>(descriptors.size());
	for (int i = 0; i < descriptors.size(); i++)
		oclDescriptors.at(i) = oclMat(descriptors.at(i));
	oclMat queryDescriptors;
	
	//Mat queryDescriptors;
	int matchIndex, imgIdx1, imgIdx2;
	DMatch match, match12, match21;
	Point2f pt1, pt2;
	float confidence1, confidence2;
	for (imgIdx1 = images.size() - 1; imgIdx1 >= 0 ; imgIdx1--)
	for (imgIdx2 = 0; imgIdx2 < imgIdx1; imgIdx2++)
	{
		//queryDescriptors = oclDescriptors.at(imgIdx1);
		//oclDescriptors.pop_back();
		//queryDescriptors = descriptors.at(i);
		//descriptors.pop_back();

		//matcher->clear();
		//matcher->add(oclDescriptors);
		//matcher->add(descriptors);

		//matcher->match(queryDescriptors, matches);

		matcher->match(oclDescriptors.at(imgIdx1), oclDescriptors.at(imgIdx2),
			matches);

		/*
		matcher->knnMatch(oclDescriptors.at(imgIdx1), oclDescriptors.at(imgIdx2),
			matches12, 2);
		matcher->knnMatch(oclDescriptors.at(imgIdx2), oclDescriptors.at(imgIdx1),
			matches21, 2);
		const float distanceThreshold = 0.75;			// TODO constant
		*/

		for (matchIndex = 0; matchIndex < matches.size(); matchIndex++)
		{
			match12 = matches.at(matchIndex);
			//match12 = matches12.at(matchIndex).at(0);
			//match21 = matches21.at(match12.trainIdx).at(0);

			/*
			// cross-check and symmetric ratio test
			if (!(match21.trainIdx == match12.queryIdx &&
				matches12.at(matchIndex).at(0).distance / matches12.at(matchIndex).at(1).distance < distanceThreshold &&
				matches21.at(matchIndex).at(0).distance / matches21.at(matchIndex).at(1).distance < distanceThreshold))

				continue;
			*/

			pt1 = keyPoints.at(imgIdx1).at(match12.queryIdx).pt;
			pt2 = keyPoints.at(imgIdx2).at(match12.trainIdx).pt;

			confidence1 = confidenceMaps.at(imgIdx1).at<float>(pt1);
			confidence2 = confidenceMaps.at(imgIdx2).at<float>(pt2);

			if (confidence1 < confidence2)
				masks.at(imgIdx1).at<uchar>(pt1) = 0;
			else
				masks.at(imgIdx2).at<uchar>(pt2) = 0;
		}
	}

	cout << "RGBDMerger::merge(): 4) merge partial reconstructions" << endl;
	// 4) merge partial reconstructions
	Mat completeReconstruction = Mat();//partialReconstructions.at(0).clone();
	Mat luminanceMap = Mat();//images.at(0).clone();
	for (int i = 0; i < partialReconstructions.size(); i++)
	{
		// handle redundant points
		partialReconstructions.at(i).setTo(Scalar::all(0), masks.at(i) - 255);	// flip binary mask

		// join partial models
		completeReconstruction.push_back(partialReconstructions.at(i));
		luminanceMap.push_back(images.at(i));
	}

	this->pointCloud = completeReconstruction;
	this->pointColors = luminanceMap;

	return completeReconstruction;
}