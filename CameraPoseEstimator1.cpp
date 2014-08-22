#include <iostream>	// for debugging
#include <string>	// for debugging
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>	// for debugging
#include "CameraPoseEstimator1.h"


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
	const bool doCrossCheck = true;
	
	Feature2D* detectorAndExtractor = new ORB();
	this->detector = detectorAndExtractor;
	this->extractor = detectorAndExtractor;
	this->matcher = new BFMatcher(NORM_HAMMING, doCrossCheck);
	
	//this->detector = new StarFeatureDetector();
	//this->detector = new FastFeatureDetector();
	//this->extractor = new FREAK();
	//this->matcher = new BFMatcher(NORM_HAMMING, doCrossCheck);
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
		//cvtColor(images.at(i), bwImages.at(i), CV_RGB2GRAY);
		//bwImages.at(i).convertTo(bwImages.at(i), CV_8UC1, 255);
		images.at(i).convertTo(bwImages.at(i), CV_8UC1, 255);
	}
	this->detector->detect(bwImages, keyPoints);
	this->extractor->compute(bwImages, keyPoints, descriptors);

	// 2) match features
	vector<vector<DMatch>> matches = vector<vector<DMatch>>(descriptors.size());
	vector<Point2f> points1, points2;
	int pointCount, queryIndex, trainIndex, validPointCount, bestPointCount;
	Mat F, E, w, u, vt, t1, t2, currentR, currentT, bestR, bestT, Rt,
		resultPoints;
	vector<Mat> R12 = vector<Mat>();
	vector<Mat> t12 = vector<Mat>();
	rotationType totalRotation = Mat::eye(3, 3, CV_64FC1);
	translationType totalTranslation = Mat(3, 1, CV_64FC1, Scalar(0));
	for (int i = 0; i < descriptors.size() - 1; i++)
	{
		// match descriptors
		this->matcher->match(descriptors.at(i), descriptors.at(i + 1),
			matches.at(i));

		// debugging
		/*
		Mat matchImg;
		drawMatches(images.at(i), keyPoints.at(i), images.at(i+1), keyPoints.at(i+1), matches.at(i), matchImg);
		string window0 = "matches " + to_string((long double) i);
		namedWindow(window0, WINDOW_AUTOSIZE);// Create a window for display. (scale down size)
		imshow(window0, matchImg);
		*/

		// compute fundamental matrix
		const int FUNDAMENTAL_MATRIX_METHOD = CV_FM_RANSAC;

		// initialize the points here ... */
		pointCount = matches.at(i).size();
		points1 = vector<Point2f>(pointCount);
		points2 = vector<Point2f>(pointCount);
		for( int j = 0; j < pointCount; j++ )
		{
			queryIndex = matches.at(i).at(j).queryIdx;
			trainIndex = matches.at(i).at(j).trainIdx;
			points1[j] = keyPoints.at(i).at(queryIndex).pt;
			points2[j] = keyPoints.at(i + 1).at(trainIndex).pt;
		}
		F = findFundamentalMat(points1, points2, FUNDAMENTAL_MATRIX_METHOD);
		cout << "matches: " << pointCount << endl;

		// compute essential matrix from fundamental matrix
		E = calibrationMatrix.t() * F * calibrationMatrix;

		// decompose essential matrix into rotation and translation
		SVD::compute(E, w, u, vt);

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
			cout << "valid point count = " << validPointCount << endl;
			if (validPointCount > bestPointCount)
			{
				bestPointCount = validPointCount;
				bestR = currentR;
				bestT = currentT;
			}
		}
		assert (bestPointCount > 0);

		// combine rotations and translations to form R and t to the first camera
		totalTranslation += totalRotation * bestT;
		totalRotation *= bestR;	// the order is important

		rotations.push_back(totalRotation.clone());
		translations.push_back(totalTranslation.clone());
	}
}
