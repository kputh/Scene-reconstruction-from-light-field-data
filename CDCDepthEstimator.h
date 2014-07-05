#pragma once

#include "mrf.h"
#include "ImageRenderer.h"
#include "DepthEstimator.h"

/**
 * Implementation of the depth estimation algorithm described by Tao, Hadap,
 * Malik and Ramamoorthi in "Depth from Combining Defocus and Correspondence
 * Using Light-Field Cameras".
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-06-27
 */
class CDCDepthEstimator :
	public DepthEstimator
{
	// parameters of the algorithm
	static const float ALPHA_MIN;
	static const float ALPHA_MAX;
	static const float ALPHA_STEP;
	static const Size DEFOCUS_WINDOW_SIZE;
	static const Size CORRESPONDENCE_WINDOW_SIZE;
	static const float LAMBDA_SOURCE[];
	static const float LAMBDA_FLAT;
	static const float LAMBDA_SMOOTH;
	static const double CONVERGENCE_FRACTION;

	// required for OpenCV's filter2D()
	static const int DDEPTH;
	static const Point WINDOW_CENTER;
	static const int BORDER_TYPE;
	static const Mat DEFOCUS_WINDOW;
	static const Mat CORRESPONDENCE_WINDOW;

	// variables for MRF propagation
	static vector<float> fsCost[2];

	typedef Vec2f fPair;
	
	ImageRenderer* renderer;

	void addAlphaData(Mat& response, float alpha);
	void calculateDefocusResponse(LightFieldPicture lightfield, Mat& response,
		float alpha);
	void calculateCorrespondenceResponse(LightFieldPicture lightfield,
		Mat& response, float alpha);
	Mat argMaxAlpha(vector<Mat> responses);
	Mat argMinAlpha(vector<Mat> responses);
	Mat calculateConfidence(Mat extrema);
	Mat getFirstExtremum(Mat extrema);
	Mat mrf(Mat depth1, Mat depth2, Mat confidence1, Mat confidence2);
	Mat pickDepthWithMaxConfidence(Mat depth1, Mat depth2, Mat confidence1,
		Mat confidence2);

	static MRF::CostVal dataCost(int pix, MRF::Label i);
	static MRF::CostVal fnCost(int pix1, int pix2, MRF::Label i, MRF::Label j);

public:
	CDCDepthEstimator(void);
	~CDCDepthEstimator(void);

	Mat estimateDepth(const LightFieldPicture lightfield);
};

