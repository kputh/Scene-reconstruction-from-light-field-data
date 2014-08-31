#include <iostream>	// for debugging
#include <string>	// for debugging
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>	// for debugging
#include "CameraPoseEstimator1.h"

using namespace ocl;

const double CameraPoseEstimator1::ZERO_THRESHOLD = 0.01;
const Mat CameraPoseEstimator1::TEST_POINTS = (Mat_<double>(4, 13) <<
	0,	0,	0,	0,	0,	0,	0,	0,		0,		0,		0,		0,		0,
	0,	0,	0,	0,	0,	0,	0,	0,		0,		0,		0,		0,		0,
	1,	2,	4,	8,	16,	32,	64,	128,	256,	512,	1024,-	2048,	4096,
	1,	1,	1,	1,	1,	1,	1,	1,		1,		1,		1,		1,		1);

const Mat CameraPoseEstimator1::R90 = (Mat_<double>(3, 3) <<
	0,	-1,	0,
	1,	0,	0,
	0,	0,	1);


CameraPoseEstimator1::CameraPoseEstimator1(void)
{
	const bool doCrossCheck = false;
	
	Feature2D* detectorAndExtractor = new ORB();
	this->detector = detectorAndExtractor;
	this->extractor = detectorAndExtractor;
	this->matcher = new BFMatcher(NORM_HAMMING, doCrossCheck);
	//this->matcher = new ocl::BruteForceMatcher_OCL_base(
	//	ocl::BruteForceMatcher_OCL_base::HammingDist);	// no cross-checking
}


CameraPoseEstimator1::~CameraPoseEstimator1(void)
{
	//delete this->detectorAndExtractor;
	delete this->detector;
	delete this->extractor;
	delete this->matcher;
}


void CameraPoseEstimator1::estimateCameraPoses(const vector<Mat>& images,
	const Mat& calibrationMatrix)
{
	cout << "CameraPoseEstimator1::estimateCameraPoses(): start" << endl;

	// initialize result vectors
	this->rotations		= vector<rotationType>();
	this->translations	= vector<translationType>();

	const int imageCount = images.size();
	vector<vector<KeyPoint>> keyPoints;//	= vector<vector<KeyPoint>>(imageCount);
	vector<Mat> descriptors				= vector<Mat>(imageCount);

	// 1) detect features and extract descriptors
	// ORB-specific image conversion
	vector<Mat> bwImages = vector<Mat>(images.size());
	for (int i = 0; i < images.size(); i++)
	{
		cvtColor(images.at(i), bwImages.at(i), CV_RGB2GRAY);
		bwImages.at(i).convertTo(bwImages.at(i), CV_8UC1, 255);
	}
	this->detector->detect(bwImages, keyPoints);
	this->extractor->compute(bwImages, keyPoints, descriptors);
	
	/*
	vector<oclMat> oclDescriptors = vector<oclMat>(descriptors.size());
	for (int i = 0; i < descriptors.size(); i++)
		oclDescriptors.at(i) = oclMat(descriptors.at(i));
	*/

	// 2) match features
	vector<DMatch> matches = vector<DMatch>();
	vector<vector<DMatch>> matches12, matches21;
	vector<DMatch> knn1, knn2;
	DMatch match12, match21;
	vector<Point2f> points1, points2;
	int pointCount, queryIndex, trainIndex, validPointCount, bestPointCount;
	Mat F, w, u, vt, t1, t2, currentR, currentT, bestR, bestT, Rt,
		resultPoints;
	vector<Mat> R12 = vector<Mat>();
	vector<Mat> t12 = vector<Mat>();
	rotationType totalRotation = Mat::eye(3, 3, CV_64FC1);
	translationType totalTranslation = Mat(3, 1, CV_64FC1, Scalar(0));
	rotations.push_back(totalRotation.clone());
	translations.push_back(totalTranslation.clone());
	int imgIdx1, imgIdx2, matchIndex;
	const float distanceThreshold = 0.75;			//TODO constant
	for (imgIdx1 = 0, imgIdx2 = 1; imgIdx1 < images.size() - 1; imgIdx1++, imgIdx2++)
	{
		// match
		matcher->knnMatch(descriptors.at(imgIdx1), descriptors.at(imgIdx2), matches12, 2);
		matcher->knnMatch(descriptors.at(imgIdx2), descriptors.at(imgIdx1), matches21, 2);
		/*
		matcher->knnMatch(oclDescriptors.at(imgIdx1), oclDescriptors.at(imgIdx2),
			matches12, 2);
		matcher->knnMatch(oclDescriptors.at(imgIdx2), oclDescriptors.at(imgIdx1),
			matches21, 2);
		*/
		// cross-check and symmetric ratio test
		matches.clear();
		for (matchIndex = 0; matchIndex < matches12.size(); matchIndex++)
		{
			knn1 = matches12.at(matchIndex);
			match12 = knn1.at(0);
			knn2 = matches21.at(match12.trainIdx);
			match21 = knn2.at(0);
			
			if (match21.trainIdx == match12.queryIdx
				&& knn1.at(0).distance / knn1.at(1).distance < distanceThreshold
				&& knn2.at(0).distance / knn2.at(1).distance < distanceThreshold
			)

			matches.push_back(match12);
		}

		if (matches.size() < 8)
		{
			matches.clear();
			for (matchIndex = 0; matchIndex < matches12.size(); matchIndex++)
			{
				knn1 = matches12.at(matchIndex);
				match12 = knn1.at(0);
				knn2 = matches21.at(match12.trainIdx);
				match21 = knn2.at(0);
			
				if (match21.trainIdx == match12.queryIdx
					&& knn1.at(0).distance / knn1.at(1).distance < distanceThreshold
					//&& knn2.at(0).distance / knn2.at(1).distance < distanceThreshold
				)

					matches.push_back(match12);
			}
		}
		
		if (matches.size() < 8)
		{
			matches.clear();
			for (matchIndex = 0; matchIndex < matches12.size(); matchIndex++)
			{
				knn1 = matches12.at(matchIndex);
				match12 = knn1.at(0);
				knn2 = matches21.at(match12.trainIdx);
				match21 = knn2.at(0);
			
				if (match21.trainIdx == match12.queryIdx
					//&& knn1.at(0).distance / knn1.at(1).distance < distanceThreshold
					//&& knn2.at(0).distance / knn2.at(1).distance < distanceThreshold
				)

					matches.push_back(match12);
			}
		}

		/*
		// debugging - render matches
		Mat matchImg;
		drawMatches(images.at(imgIdx1), keyPoints.at(imgIdx1),
			images.at(imgIdx2), keyPoints.at(imgIdx2), matches, matchImg);
		string window0 = "matches " + to_string((long double) imgIdx1);
		namedWindow(window0, WINDOW_AUTOSIZE);// Create a window for display. (scale down size)
		imshow(window0, matchImg);

		waitKey(0);
		*/

		assert (matches.size() >= 8);

		// compute fundamental matrix
		const int FUNDAMENTAL_MATRIX_METHOD = CV_FM_RANSAC;

		// initialize the points here ... */
		pointCount = matches.size();
		points1 = vector<Point2f>(pointCount);
		points2 = vector<Point2f>(pointCount);
		for( int j = 0; j < pointCount; j++ )
		{
			queryIndex = matches.at(j).queryIdx;
			trainIndex = matches.at(j).trainIdx;
			points1[j] = keyPoints.at(imgIdx1).at(queryIndex).pt;
			points2[j] = keyPoints.at(imgIdx2).at(trainIndex).pt;
		}
		F = findFundamentalMat(points1, points2, FUNDAMENTAL_MATRIX_METHOD);

		// decompose fundamental matrix into rotation and translation
		SVD::compute(F, w, u, vt);

		t12.clear(); t12.push_back(u.col(2)); t12.push_back(-u.col(2));

		R12.clear();
		Mat RCandidate = u * R90.t() * vt;
		if (abs(determinant(RCandidate) - 1.) < ZERO_THRESHOLD)
			R12.push_back(RCandidate);
		RCandidate = u * R90 * vt;
		if (abs(determinant(RCandidate) - 1.) < ZERO_THRESHOLD)
			R12.push_back(RCandidate);
		RCandidate = -u * R90.t() * vt;
		if (abs(determinant(RCandidate) - 1.) < ZERO_THRESHOLD)
			R12.push_back(RCandidate);
		RCandidate = -u * R90 * vt;
		if (abs(determinant(RCandidate) - 1.) < ZERO_THRESHOLD)
			R12.push_back(RCandidate);

		assert (R12.size() == 2);

		// pick the one valid combination of R and t
		bestPointCount = -1;
		for (int rI = 0; rI < 2; rI++)
		for (int tI = 0; tI < 2; tI++)
		{
			currentR = R12.at(rI);	currentT = t12.at(tI);
			hconcat(currentR, currentT, Rt);
			resultPoints = Rt * TEST_POINTS;

			resultPoints.convertTo(resultPoints, CV_32FC1);
			threshold(resultPoints.row(2), resultPoints, ZERO_THRESHOLD, 1.0,
				THRESH_BINARY);
			validPointCount = countNonZero(resultPoints);
			if (validPointCount > bestPointCount)
			{
				bestPointCount = validPointCount;
				bestR = currentR;
				bestT = currentT;
			}
		}
		assert (bestPointCount > 0);

		printf("CameraPoseEstimator1: %i points in view for pose %i\n",
			validPointCount, imgIdx1);

		// combine rotations and translations to form R and t to the first camera
		totalTranslation += totalRotation * bestT;
		totalRotation *= bestR;	// the order is important

		rotations.push_back(totalRotation.clone());
		translations.push_back(totalTranslation.clone());
	}

	assert (rotations.size() == images.size());
	assert (translations.size() == images.size());

	cout << "CameraPoseEstimator1::estimateCameraPoses(): end" << endl;
}
