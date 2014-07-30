#include <opencv2\calib3d\calib3d.hpp>
#include "CameraPoseEstimator1.h"


CameraPoseEstimator1::CameraPoseEstimator1(void)
{
	const bool doCrossCheck = true;

	Feature2D* detectorAndExtractor = new ORB();
	this->detector = detectorAndExtractor;
	this ->extractor = detectorAndExtractor;
	//this->detectorAndExtractor = new ORB();
	this->matcher = new BFMatcher(NORM_HAMMING, doCrossCheck);
}


CameraPoseEstimator1::~CameraPoseEstimator1(void)
{
	//delete this->detectorAndExtractor;
	delete this->matcher;
}


void CameraPoseEstimator1::estimateCameraPoses(const vector<Mat>& images,
	const Mat& calibrationMatrix) const
{
	vector<vector<KeyPoint>> keyPoints = vector<vector<KeyPoint>>();
	vector<Mat> descriptors = vector<Mat>();

	// 1) detect features and extract descriptors
	this->detector->detect(images, keyPoints);
	this->extractor->compute(images, keyPoints, descriptors);

	// 2) match features
	/*
	const int numberOfSetMatchings = descriptors.size() - 1;
	Mat queryDescriptors;
	vector<Mat> trainDescriptors = vector<Mat>(descriptors);
	vector<vector<DMatch>> matches = vector<vector<DMatch>>(numberOfSetMatchings);
	for (int i = numberOfSetMatchings - 1; i >= 0; i--)
	{
		queryDescriptors = trainDescriptors.back();
		trainDescriptors.pop_back();

		this->matcher->clear();
		this->matcher->add(trainDescriptors);

		this->matcher->match(queryDescriptors, matches.at(i));
	}
	*/
	vector<vector<DMatch>> matches = vector<vector<DMatch>>(descriptors.size());
	vector<Point2f> points1, points2;
	Mat totalRotation = Mat::eye(3, 3, CV_32FC1);
	Vec4f totalTranslation = Vec4f(0, 0, 0, 1);
	vector<Mat> rotations = vector<Mat>();
	vector<Vec4f> translations = vector<Vec4f>();
	for (int i = 0; i < descriptors.size() - 1; i++)
	{
		// match descriptors
		this->matcher->match(descriptors.at(i), descriptors.at(i + 1),
			matches.at(i));

		// compute fundamental matrix
		const int FUNDAMENTAL_MATRIX_METHOD = CV_FM_RANSAC;

		// initialize the points here ... */
		int point_count = matches.at(i).size();
		points1 = vector<Point2f>(point_count);
		points2 = vector<Point2f>(point_count);
		for( int j = 0; j < point_count; j++ )
		{
			int queryIndex = matches.at(i).at(j).queryIdx;
			int trainIndex = matches.at(i).at(j).trainIdx;
			points1[j] = keyPoints.at(i).at(queryIndex).pt;
			points2[j] = keyPoints.at(i + 1).at(trainIndex).pt;
		}
		Mat F = findFundamentalMat(points1, points2, FUNDAMENTAL_MATRIX_METHOD);

		// compute essential matrix from fundamental matrix
		Mat E = calibrationMatrix.t() * F * calibrationMatrix;

		// decompose essential matrix into rotation and translation
		Mat cameraMatrix, rotation;
		Vec4f translation;
		decomposeProjectionMatrix(E, cameraMatrix, rotation, translation);

		// combine rotations and translations to for R and t to the first camera
		composeRT(rotation, translation, totalRotation, totalTranslation,
			 totalRotation, totalTranslation);

		 rotations.push_back(totalRotation);
		 translations.push_back(totalTranslation);
	}

	// TODO
}
