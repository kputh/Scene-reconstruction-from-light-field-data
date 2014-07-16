#pragma once

#include "mrf.h"
#include "ImageRenderer1.h"
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
	static const int DEPTH_RESOLUTION;
	static const Size DEFOCUS_WINDOW_SIZE;
	static const Size CORRESPONDENCE_WINDOW_SIZE;
	static const float LAMBDA_SOURCE[];
	static const float LAMBDA_FLAT;
	static const float LAMBDA_SMOOTH;
	static const double CONVERGENCE_FRACTION;

	// required for OpenCV's filter2D(), used for averaging over a window
	static const int DDEPTH;
	static const Point WINDOW_CENTER;
	static const int BORDER_TYPE;
	static const Mat DEFOCUS_WINDOW;
	static const Mat CORRESPONDENCE_WINDOW;

	// used for defocus response calculation
	static const int LAPLACIAN_KERNEL_SIZE;
	static const Mat LoG;

	static const int MAT_TYPE;

	typedef Vec2f fPair;
	
	ImageRenderer1* renderer;

	Size imageSize;
	Vec2f angularCorrection;
	Vec2f fromCornerToCenter;
	int Nuv;

	oclMat depthMap;
	oclMat confidenceMap;
	oclMat extendedDepthOfFieldImage;

	oclMat calculateDefocusResponse(const LightFieldPicture& lightfield,
		const oclMat& refocusedImage, const float alpha);
	oclMat calculateCorrespondenceResponse(const LightFieldPicture& lightfield,
		const oclMat& refocusedImage, const float alpha);
	void normalizeConfidence(oclMat& confidence1, oclMat& confidence2);
	oclMat mrf(const oclMat& depth1, const oclMat& depth2,
		const oclMat& confidence1, const oclMat& confidence2);
	Mat pickDepthWithMaxConfidence(Mat& depth1, Mat& depth2,
		Mat& confidence1, Mat& confidence2);

	static MRF::CostVal dataCost(int pix, MRF::Label i);
	static MRF::CostVal fnCost(int pix1, int pix2, MRF::Label i, MRF::Label j);

public:
	CDCDepthEstimator(void);
	~CDCDepthEstimator(void);

	oclMat estimateDepth(const LightFieldPicture& lightfield);
	oclMat getDepthMap() const;
	oclMat getConfidenceMap() const;
	oclMat getExtendedDepthOfFieldImage() const;
};

