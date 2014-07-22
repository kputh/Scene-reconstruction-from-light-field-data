#include <iostream>	// debugging
#include <cfloat>	// debugging
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>	// debugging
#include "mrf.h"
#include "MaxProdBP.h"
#include "BP-S.h"	// does it work better than BP-M?
#include "ImageRenderer1.h"
#include "CDCDepthEstimator.h"
#include "Util.h"


const float CDCDepthEstimator::ALPHA_MIN					= 0.2;
const float CDCDepthEstimator::ALPHA_MAX					= 2.0;
const int CDCDepthEstimator::DEPTH_RESOLUTION				= 25;
const Size CDCDepthEstimator::DEFOCUS_WINDOW_SIZE			= Size(9, 9);
const Size CDCDepthEstimator::CORRESPONDENCE_WINDOW_SIZE	= Size(9, 9);
const float CDCDepthEstimator::LAMBDA_SOURCE[]				= { 1, 1 };
const float CDCDepthEstimator::LAMBDA_FLAT					= 2;
const float CDCDepthEstimator::LAMBDA_SMOOTH				= 2;
const double CDCDepthEstimator::CONVERGENCE_FRACTION		= 1;

const int CDCDepthEstimator::DDEPTH					= -1;
const Point CDCDepthEstimator::WINDOW_CENTER		= Point (-1, -1);
const int CDCDepthEstimator::BORDER_TYPE			= BORDER_REPLICATE;
const Mat CDCDepthEstimator::DEFOCUS_WINDOW
	= Mat(DEFOCUS_WINDOW_SIZE, CV_32FC1,
	Scalar(1 / (float) DEFOCUS_WINDOW_SIZE.area()));
const Mat CDCDepthEstimator::CORRESPONDENCE_WINDOW
	= Mat(CORRESPONDENCE_WINDOW_SIZE, CV_32FC1,
	Scalar(1 / (float) CORRESPONDENCE_WINDOW_SIZE.area()));

const int CDCDepthEstimator::LAPLACIAN_KERNEL_SIZE = 9;
const Mat CDCDepthEstimator::LoG = (
	Mat_<float>(CDCDepthEstimator::LAPLACIAN_KERNEL_SIZE,
	CDCDepthEstimator::LAPLACIAN_KERNEL_SIZE) << 
	0,	-1,	-1,	-2,	-2,	-2,	-1,	-1,	0,
	-1,	-2, -4, -5, -5, -5,	-4, -2,	-1,
	-1,	-4,	-5,	-3,	0,	-3,	-5,	-4,	-1,
	-2,	-5,	-3,	12,	24,	12,	-3,	-5,	-2,
	-2,	-5,	0,	24, 40, 24,	0,	-5,	-2,
	-2,	-5,	-3,	12,	24,	12,	-3,	-5,	-2,
	-1,	-4,	-5,	-3,	0,	-3,	-5,	-4,	-1,
	-1,	-2, -4, -5, -5, -5,	-4, -2,	-1,
	0,	-1,	-1,	-2,	-2,	-2,	-1,	-1,	0);


const int CDCDepthEstimator::MAT_TYPE = CV_32FC1;

// used for MRF propagation
vector<MRF::CostVal> CDCDepthEstimator::dataCost1;
vector<MRF::CostVal> CDCDepthEstimator::dataCost2;
vector<MRF::CostVal> CDCDepthEstimator::fsCost1;
vector<MRF::CostVal> CDCDepthEstimator::fsCost2;


CDCDepthEstimator::CDCDepthEstimator(void)
{
	this->renderer	=  new ImageRenderer1;
}


CDCDepthEstimator::~CDCDepthEstimator(void)
{
}


oclMat CDCDepthEstimator::estimateDepth(const LightFieldPicture& lightfield)
{
	// 1) for each shear, compute depth response
	// also compute "running" response extrema, depth map and extended depth of
	// field image
	this->renderer->setLightfield(lightfield);
	this->imageSize	= this->renderer->imageSize;
	this->angularCorrection = Vec2f(lightfield.ANGULAR_RESOLUTION.width, 
		lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	this->NuvMultiplier	= 1. / (double) lightfield.ANGULAR_RESOLUTION.area();

	// used for cropping subaperture image to image size
	const int srcWidth		= lightfield.SPARTIAL_RESOLUTION.width;
	const int srcHeight		= lightfield.SPARTIAL_RESOLUTION.height;
	const int left			= (imageSize.width - srcWidth) / 2;
	const int top			= (imageSize.height - srcHeight) / 2;
	this->fromCornerToCenter	= Vec2f(left, top);

	const float alphaStep = (ALPHA_MAX - ALPHA_MIN) / (float) DEPTH_RESOLUTION;
	oclMat refocusedImage, response, maxDefocusResponse, max2, minCorrespondenceResponse, min2,
		dEDOF, cEDOF, defocusAlpha, correspondenceAlpha, mask1, mask2;

	float alpha = ALPHA_MIN;
	Scalar scalarAlpha = Scalar(alpha);
	defocusAlpha = oclMat(imageSize, MAT_TYPE);
	defocusAlpha.setTo(scalarAlpha);
	correspondenceAlpha = oclMat(imageSize, MAT_TYPE);
	correspondenceAlpha.setTo(scalarAlpha);

	this->renderer->setAlpha(alpha);
	refocusedImage = this->renderer->renderImage();
	refocusedImage.copyTo(dEDOF);
	refocusedImage.copyTo(cEDOF);

	response = calculateDefocusResponse(lightfield, refocusedImage, alpha);
	response.copyTo(maxDefocusResponse);
	response.copyTo(max2);

	response = calculateCorrespondenceResponse(lightfield, refocusedImage, alpha);
	response.copyTo(minCorrespondenceResponse);
	response.copyTo(min2);
	
	for (alpha = ALPHA_MIN + alphaStep; alpha <= ALPHA_MAX; alpha += alphaStep)
	{
		scalarAlpha = Scalar(alpha);
		this->renderer->setAlpha(alpha);
		refocusedImage = this->renderer->renderImage();


		// handle defocus-based algorithm
		response = calculateDefocusResponse(lightfield, refocusedImage, alpha);

		// find first maximum
		mask1 = (response > maxDefocusResponse);

		// find second maximum
		mask2 = (response < maxDefocusResponse) & (response > max2);

		// update maxima
		response.copyTo(maxDefocusResponse, mask1);
		response.copyTo(max2, mask2);

		// update depth estimation
		defocusAlpha.setTo(scalarAlpha, mask1);

		// update extended depth of field image
		refocusedImage.copyTo(dEDOF, mask1);


		// handle correspondence-based algorithm
		response = calculateCorrespondenceResponse(lightfield, refocusedImage, alpha);

		// find first minimum
		mask1 = (response < minCorrespondenceResponse);

		// find second minimum
		mask2 = (response > minCorrespondenceResponse) & (response < min2);

		// update minima
		response.copyTo(minCorrespondenceResponse, mask1);
		response.copyTo(min2, mask2);

		// update depth estimation
		correspondenceAlpha.setTo(scalarAlpha, mask1);

		// update extended depth of field image
		refocusedImage.copyTo(cEDOF, mask1);
	}
	oclMat defocusConfidence, correspondenceConfidence;
	ocl::divide(maxDefocusResponse, max2, defocusConfidence);
	ocl::divide(min2, minCorrespondenceResponse, correspondenceConfidence); // laut Paper umgekehrt

	// normalize confidence (from MatLab code)
	//normalizeConfidence(defocusConfidence, correspondenceConfidence);

	// 3) global operation to combine cues
	oclMat labels = mrf(defocusAlpha, correspondenceAlpha,
		defocusConfidence, correspondenceConfidence);
	
	/*
	oclMat labels = pickLabelWithMaxConfidence(defocusConfidence,
		correspondenceConfidence);
	*/

	// translate label map into depth map
	oclMat alphaMap, confidenceMap, extendedDepthOfFieldImage;

	mask1 = (labels == 0);
	defocusAlpha.copyTo(alphaMap, mask1);
	defocusConfidence.copyTo(confidenceMap, mask1);
	dEDOF.copyTo(extendedDepthOfFieldImage, mask1);

	mask1 = (labels == 1);
	correspondenceAlpha.copyTo(alphaMap, mask1);
	correspondenceConfidence.copyTo(confidenceMap, mask1);
	cEDOF.copyTo(extendedDepthOfFieldImage, mask1);

	// 4) compute actual depth from alpha values
	oclMat depthMap;
	ocl::multiply(lightfield.getRawFocalLength(), alphaMap, depthMap);

	// lens equation:		1/f = 1/d_obj + 1/d_img
	// derived equation:	d_obj = (f * d_img) / (d_img - f)
	// d_img is unknown
	// TODO

	// debugging
	/*
	//renderer->setFocalLength(?);
	oclMat image = renderer->renderImage();

	Mat m;
	
	defocusAlpha.download(m); normalize(m,m,0,1,NORM_MINMAX);
	string window1 = "depth (alpha) from defocus";
	namedWindow(window1, WINDOW_NORMAL);
	imshow(window1, m);
	
	correspondenceAlpha.download(m); normalize(m,m,0,1,NORM_MINMAX);
	string window2 = "depth (alpha) from correspondence";
	namedWindow(window2, WINDOW_NORMAL);
	imshow(window2, m);
	
	defocusConfidence.download(m);
	//normalize(m,m,0,1,NORM_MINMAX);
	string window3 = "confidence from defocus";
	namedWindow(window3, WINDOW_NORMAL);
	imshow(window3, m);
	
	correspondenceConfidence.download(m);
	//normalize(m, m, 0, 1, NORM_MINMAX);
	string window4 = "confidence from correspondence";
	namedWindow(window4, WINDOW_NORMAL);
	imshow(window4, m);
	
	image.download(m);
	string window5 = "central perspective";
	namedWindow(window5, WINDOW_NORMAL);
	imshow(window5, m);
	
	alphaMap.download(m); normalize(m,m,0,1,NORM_MINMAX);
	string window6 = "combined alpha map";
	namedWindow(window6, WINDOW_NORMAL);
	imshow(window6, m);
	/*
	confidenceMap.download(m); normalize(m,m,0,1,NORM_MINMAX);
	string window7 = "combined confidence map";
	namedWindow(window7, WINDOW_NORMAL);
	imshow(window7, m);
	
	extendedDepthOfFieldImage.download(m);
	string window8 = "extended Depth Of Field Image";
	namedWindow(window8, WINDOW_NORMAL);
	imshow(window8, m);
	
	labels.download(m); m.convertTo(m, CV_32FC1);
	string window9 = "labels";
	namedWindow(window9, WINDOW_NORMAL);
	imshow(window9, m);
	
	waitKey(0);
	*/
	

	// crop all resulting Mats
	const int width = lightfield.SPARTIAL_RESOLUTION.width;
	const int height = lightfield.SPARTIAL_RESOLUTION.height;
	const int cropLeft = (depthMap.size().width - width) / 2;
	const int cropTop = (depthMap.size().height - height) / 2;
	const Rect cropRect = Rect(cropLeft, cropTop, width, height);

	oclMat oclm;
	oclm = oclMat(depthMap, cropRect);
	oclm.copyTo(depthMap);
	oclm = oclMat(confidenceMap, cropRect);
	oclm.copyTo(confidenceMap);
	oclm = oclMat(extendedDepthOfFieldImage, cropRect);
	oclm.copyTo(extendedDepthOfFieldImage);

	this->depthMap					= depthMap;
	this->confidenceMap				= confidenceMap;
	this->extendedDepthOfFieldImage	= extendedDepthOfFieldImage;

	return depthMap;
}


oclMat CDCDepthEstimator::calculateDefocusResponse(
	const LightFieldPicture& lightfield, const oclMat& refocusedImage,
	const float alpha)
{
	oclMat response, d2x, d2y;

	//ocl::Laplacian(refocusedImage, response, CV_32F, LAPLACIAN_KERNEL_SIZE); // Größe wird nicht unterstützt
	
	//ocl::Scharr(refocusedImage, d2x, CV_32FC1, 1, 0, LAPLACIAN_KERNEL_SIZE);
	//ocl::Scharr(refocusedImage, d2y, CV_32FC1, 0, 1, LAPLACIAN_KERNEL_SIZE);
	ocl::Sobel(refocusedImage, d2x, CV_32FC1, 2, 0, LAPLACIAN_KERNEL_SIZE);
	ocl::Sobel(refocusedImage, d2y, CV_32FC1, 0, 2, LAPLACIAN_KERNEL_SIZE);
	response = d2x + d2y;

	/*
	ocl::multiply(d2x, d2x, d2x);
	ocl::multiply(d2y, d2y, d2y);
	ocl::add(d2x, d2y, response);
	ocl::pow(response, 0.5, response);
	*/

	/*
	ocl::filter2D(refocusedImage, response, DDEPTH, LoG,
		WINDOW_CENTER, 0, BORDER_TYPE);
	*/

	ocl::abs(response, response);
	ocl::filter2D(response, response, DDEPTH, DEFOCUS_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	return response;
}


oclMat CDCDepthEstimator::calculateCorrespondenceResponse(
	const LightFieldPicture& lightfield, const oclMat& refocusedImage,
	const float alpha)
{
	const float weight = 1. - 1. / alpha;

	oclMat subapertureImage, modifiedSubapertureImage, differenceImage,
		squaredDifference;
	oclMat variance = oclMat(imageSize, CV_32FC1, Scalar(0));
	Vec2f translation;
	Point2f dstTri[3];
	Mat transformation;

	int u, v;
	for (v = 0; v < lightfield.ANGULAR_RESOLUTION.height; v++)
		for (u = 0; u < lightfield.ANGULAR_RESOLUTION.width; u++)
		{
			// get subaperture image
			subapertureImage = lightfield.getSubapertureImageI(u, v);

			// translate and crop subaperture image
			translation = (Vec2f(u, v) - angularCorrection) * weight;
			translation += fromCornerToCenter;
			dstTri[0] = Point2f(0 + translation[0], 0 + translation[1]);
			dstTri[1] = Point2f(1 + translation[0], 0 + translation[1]);
			dstTri[2] = Point2f(0 + translation[0], 1 + translation[1]);
			transformation = getAffineTransform(UNIT_VECTORS, dstTri);

			ocl::warpAffine(subapertureImage, modifiedSubapertureImage,
				transformation, imageSize, INTER_LINEAR);

			// compute response
			differenceImage = modifiedSubapertureImage - refocusedImage;
			ocl::multiply(differenceImage, differenceImage, squaredDifference);
			ocl::add(squaredDifference, variance, variance);
		}

	ocl::multiply(NuvMultiplier, variance, variance);

	oclMat standardDeviation, confidence;
	ocl::pow(variance, 0.5, standardDeviation);	// es gibt kein ocl::sqrt()
	ocl::filter2D(standardDeviation, confidence, DDEPTH, CORRESPONDENCE_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	/*
	ocl::filter2D(variance, confidence, DDEPTH, CORRESPONDENCE_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	*/
	return confidence;
}


void CDCDepthEstimator::normalizeConfidence(oclMat& confidence1, oclMat& confidence2)
{
	// find greatest confidence in both matrices combined
	oclMat maxConfidenceMat;
	double minVal, maxVal;

	ocl::max(confidence1, confidence2, maxConfidenceMat);
	ocl::minMax(maxConfidenceMat, &minVal, &maxVal);

	double multiplier = 1. / maxVal;
	ocl::multiply(multiplier, confidence1, confidence1);
	ocl::multiply(multiplier, confidence2, confidence2);
}


oclMat CDCDepthEstimator::pickLabelWithMaxConfidence(const oclMat& confidence1,
	const oclMat& confidence2) const
{
	oclMat labels = oclMat(confidence1.size(), CV_8UC1, Scalar(0));

	oclMat mask;
	mask = confidence1 < confidence2;
	labels.setTo(Scalar(1), mask);

	return labels;
}


MRF::CostVal CDCDepthEstimator::dataCost(int pix, MRF::Label i)
{
	//return 0;

	switch (i)
	{
	case 0:
		return CDCDepthEstimator::dataCost1[pix];
		//return CDCDepthEstimator::dataCost1[pix] + CDCDepthEstimator::fsCost1[pix];
	case 1:
		return CDCDepthEstimator::dataCost2[pix];
		//return CDCDepthEstimator::dataCost2[pix] + CDCDepthEstimator::fsCost2[pix];
	default:
		return 0;
	}
}


MRF::CostVal CDCDepthEstimator::fnCost(int pix1, int pix2,
	MRF::Label i, MRF::Label j)
{
	return 0;

	/*
	if (pix1 > pix2)
	{
		int tmp = pix1; pix1 = pix2; pix2 = tmp;
		tmp = i; i = j; j = tmp;
	}
	*/
	/*
	if (pix1 == pix2)
		return 0;
	*/

	switch (j)
	{
	case 0:
		return CDCDepthEstimator::fsCost1[pix2];
	case 1:
		return CDCDepthEstimator::fsCost2[pix2];
	default:
		return 0;
	}
}


oclMat CDCDepthEstimator::mrf(const oclMat& depth1, const oclMat& depth2,
	const oclMat& confidence1, const oclMat& confidence2)
{
	MRF* mrf;
	EnergyFunction *energy;
	float time;

	const int ENERGY_KERNEL_SIZE = 3;

	// pre-calculate cost
	Mat tmpMat;
	oclMat aDiffs, gradientX, gradientY, laplacian, dataCost, flatnessCost,
		smoothnessCost, totalCost;
	ocl::absdiff(depth1, depth2, aDiffs);

	// calculate cost for defocus solution
	ocl::multiply(aDiffs, confidence2, dataCost);
	ocl::multiply(LAMBDA_SOURCE[0], dataCost, dataCost);

	ocl::Sobel(depth1, gradientX, CV_32FC1, 1, 0, ENERGY_KERNEL_SIZE);
	ocl::Sobel(depth1, gradientY, CV_32FC1, 0, 1, ENERGY_KERNEL_SIZE);
	ocl::abs(gradientX, gradientX);
	ocl::abs(gradientY, gradientY);
	ocl::add(gradientX, gradientY, flatnessCost);

	ocl::Sobel(depth1, gradientX, CV_32FC1, 2, 0, ENERGY_KERNEL_SIZE);
	ocl::Sobel(depth1, gradientY, CV_32FC1, 0, 2, ENERGY_KERNEL_SIZE);
	ocl::add(gradientX, gradientY, laplacian);
	ocl::abs(laplacian, laplacian);
	ocl::multiply(LAMBDA_SMOOTH, laplacian, smoothnessCost);

	oclMat fsCost;
	ocl::add(flatnessCost, smoothnessCost, fsCost);
	Mat filter = Mat(3, 3, CV_32FC1, Scalar(1));
	ocl::filter2D(fsCost, fsCost, CV_32FC1, filter);
	ocl::add(dataCost, fsCost, totalCost);

	/*
	ocl::add(dataCost, flatnessCost, totalCost);
	ocl::add(totalCost, smoothnessCost, totalCost);
	//totalCost = dataCost + flatnessCost + smoothnessCost;
	totalCost.download(tmpMat);
	tmpMat.reshape(1, 1).copyTo(CDCDepthEstimator::dataCost1);
	*/
	totalCost.download(tmpMat);
	tmpMat.reshape(1, 1).copyTo(CDCDepthEstimator::dataCost1);
	ocl::add(flatnessCost, smoothnessCost, totalCost);
	totalCost.download(tmpMat);
	tmpMat.reshape(1, 1).copyTo(CDCDepthEstimator::fsCost1);

	// debugging
	double maxVal, minVal;
	oclMat maxMat, defocusDataCost, defocusFlatnessCost, defocusSmoothnessCost,
		defocusTotalCost;
	ocl::max(flatnessCost, smoothnessCost, maxMat);
	ocl::max(dataCost, maxMat, maxMat);
	dataCost.copyTo(defocusDataCost);
	flatnessCost.copyTo(defocusFlatnessCost);
	totalCost.copyTo(defocusTotalCost);
	smoothnessCost.copyTo(defocusSmoothnessCost);


	// calculate cost for corresponence solution
	ocl::multiply(aDiffs, confidence1, dataCost);
	ocl::multiply(LAMBDA_SOURCE[1], dataCost, dataCost);

	ocl::Sobel(depth2, gradientX, CV_32FC1, 1, 0, ENERGY_KERNEL_SIZE);
	ocl::Sobel(depth2, gradientY, CV_32FC1, 0, 1, ENERGY_KERNEL_SIZE);
	ocl::abs(gradientX, gradientX);
	ocl::abs(gradientY, gradientY);
	ocl::add(gradientX, gradientY, flatnessCost);

	ocl::Sobel(depth2, gradientX, CV_32FC1, 2, 0, ENERGY_KERNEL_SIZE);
	ocl::Sobel(depth2, gradientY, CV_32FC1, 0, 2, ENERGY_KERNEL_SIZE);
	ocl::add(gradientX, gradientY, laplacian);
	ocl::abs(laplacian, laplacian);
	ocl::multiply(LAMBDA_SMOOTH, laplacian, smoothnessCost);

	ocl::add(flatnessCost, smoothnessCost, fsCost);
	ocl::filter2D(fsCost, fsCost, CV_32FC1, filter);
	ocl::add(dataCost, fsCost, totalCost);

	/*
	ocl::add(dataCost, flatnessCost, totalCost);
	ocl::add(totalCost, smoothnessCost, totalCost);
	//totalCost = dataCost + flatnessCost + smoothnessCost;
	totalCost.download(tmpMat);
	tmpMat.reshape(1, 1).copyTo(CDCDepthEstimator::dataCost2);
	*/

	totalCost.download(tmpMat);
	tmpMat.reshape(1, 1).copyTo(CDCDepthEstimator::dataCost2);
	ocl::add(flatnessCost, smoothnessCost, totalCost);
	totalCost.download(tmpMat);
	tmpMat.reshape(1, 1).copyTo(CDCDepthEstimator::fsCost2);

	// debugging
	/*
	ocl::max(flatnessCost, maxMat, maxMat);
	ocl::max(smoothnessCost, maxMat, maxMat);
	ocl::max(dataCost, maxMat, maxMat);
	ocl::minMaxLoc(maxMat, &minVal, &maxVal);

	double multiplier = 1. / maxVal;
	ocl::multiply(multiplier, defocusDataCost, defocusDataCost);
	ocl::multiply(multiplier, defocusSmoothnessCost, defocusSmoothnessCost);
	ocl::multiply(multiplier, defocusFlatnessCost, defocusFlatnessCost);
	ocl::multiply(multiplier, dataCost, dataCost);
	ocl::multiply(multiplier, smoothnessCost, smoothnessCost);
	ocl::multiply(multiplier, flatnessCost, flatnessCost);

	Mat m;
	//defocusDataCost.download(m);
	//string window1 = "norm. data cost for defocus";
	defocusTotalCost.download(m);
	string window1 = "norm. total cost for defocus";
	namedWindow(window1, WINDOW_NORMAL);
	imshow(window1, m);
	
	defocusSmoothnessCost.download(m);
	string window2 = "norm. smoothness cost for defocus";
	namedWindow(window2, WINDOW_NORMAL);
	imshow(window2, m);

	defocusFlatnessCost.download(m);
	string window3 = "norm. flatness cost for defocus";
	namedWindow(window3, WINDOW_NORMAL);
	imshow(window3, m);
	
	//dataCost.download(m);
	//string window4 = "norm. data cost for correspondence";
	totalCost.download(m);
	string window4 = "norm. total cost for correspondence";
	namedWindow(window4, WINDOW_NORMAL);
	imshow(window4, m);
	
	smoothnessCost.download(m);
	string window5 = "norm. smoothness cost for correspondence";
	namedWindow(window5, WINDOW_NORMAL);
	imshow(window5, m);

	flatnessCost.download(m);
	string window6 = "norm. flatness cost for correspondence";
	namedWindow(window6, WINDOW_NORMAL);
	imshow(window6, m);
	
	ocl::max(depth1, depth2, maxMat);
	ocl::minMaxLoc(maxMat, &minVal, &maxVal);
	multiplier = 1. / maxVal;

	depth1.download(m); m *= multiplier;
	string window7 = "depth from defocus";
	namedWindow(window7, WINDOW_NORMAL);
	imshow(window7, m);

	depth2.download(m); m *= multiplier;
	string window8 = "depth from correspondence";
	namedWindow(window8, WINDOW_NORMAL);
	imshow(window8, m);

	ocl::max(confidence1, confidence2, maxMat);
	ocl::minMaxLoc(maxMat, &minVal, &maxVal);
	multiplier = 1. / maxVal;

	confidence1.download(m); m *= multiplier;
	string window9 = "confidence from defocus";
	namedWindow(window9, WINDOW_NORMAL);
	imshow(window9, m);

	confidence2.download(m); m *= multiplier;
	string window10 = "confidence from correspondence";
	namedWindow(window10, WINDOW_NORMAL);
	imshow(window10, m);

	//waitKey(0);
	*/

	// define/generate complete cost function
	DataCost *data         = new DataCost(&CDCDepthEstimator::dataCost);
	SmoothnessCost *smooth = new SmoothnessCost(&CDCDepthEstimator::fnCost);
	energy = new EnergyFunction(data, smooth);

	// compute optimized depth map (labeling)
	//mrf = new MaxProdBP(depth1.size().width, depth1.size().height, 2, energy);
	mrf = new BPS(depth1.size().width, depth1.size().height, 2, energy);
	mrf->initialize();
	mrf->clearAnswer();
	
	// debugging
	printf("Energy at the Start = %g (Es %g + Ed %g)\n", (float)mrf->totalEnergy(),
	   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());
	
	// perform optimization
	const int labelMatType = CV_32SC1;
	MRF::Label* labelsArray = mrf->getAnswerPtr();
	oclMat newLabels, tmpOclMat;
	oclMat oldLabels = oclMat(depth1.size(), labelMatType, Scalar(2));

	double rootMeanSquareDeviation;
	int pixelCount = depth1.size().area();
	do {
		// perform more optimization
		mrf->optimize(1, time);	// TODO use constant

		// calculate root-mean-square deviation
		newLabels = oclMat(depth1.size(), labelMatType, labelsArray);
		tmpOclMat = newLabels - oldLabels;
		ocl::multiply(tmpOclMat, tmpOclMat, tmpOclMat);
		rootMeanSquareDeviation = std::sqrt(ocl::sum(tmpOclMat)[0] /
			(double) pixelCount);
		
		newLabels.copyTo(oldLabels);
				
		// debugging
		printf("Current energy = %g (Es %g + Ed %g)\n", (float)mrf->totalEnergy(),
			(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());
		
	} while (rootMeanSquareDeviation > CONVERGENCE_FRACTION);

	delete mrf;

	return newLabels;
}


oclMat CDCDepthEstimator::getDepthMap() const
{
	return this->depthMap;
}


oclMat CDCDepthEstimator::getConfidenceMap() const
{
	return this->confidenceMap;
}


oclMat CDCDepthEstimator::getExtendedDepthOfFieldImage() const
{
	return this->extendedDepthOfFieldImage;
}
