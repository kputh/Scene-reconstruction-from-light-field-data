#include <iostream>	// debugging
#include <cfloat>	// debugging
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>	// debugging
#include "mrf.h"
#include "MaxProdBP.h"
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
	= Mat(DEFOCUS_WINDOW_SIZE, CV_32F,
	Scalar(1 / (float) DEFOCUS_WINDOW_SIZE.area()));
const Mat CDCDepthEstimator::CORRESPONDENCE_WINDOW
	= Mat(CORRESPONDENCE_WINDOW_SIZE, CV_32F,
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


CDCDepthEstimator::CDCDepthEstimator(void)
{
	this->renderer	=  new ImageRenderer1;
}


CDCDepthEstimator::~CDCDepthEstimator(void)
{
}


oclMat CDCDepthEstimator::estimateDepth(const LightFieldPicture& lightfield)
{
	cout << "Schritt 1+2" << endl;
	// 1) for each shear, compute depth response
	// also compute "running" response extrema, depth map and extended depth of
	// field image
	this->renderer->setLightfield(lightfield);
	this->imageSize	= this->renderer->imageSize;
	this->angularCorrection = Vec2f(lightfield.ANGULAR_RESOLUTION.width, 
		lightfield.ANGULAR_RESOLUTION.height) * 0.5;
	this->Nuv = lightfield.ANGULAR_RESOLUTION.area();

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

	oclMat defocusConfidence		= maxDefocusResponse / max2;
	oclMat correspondenceConfidence	= min2 / minCorrespondenceResponse; // / min2;

	// normalize confidence
	normalizeConfidence(defocusConfidence, correspondenceConfidence);

	cout << "Schritt 3" << endl;
	// 3) global operation to combine cues
	oclMat labels = mrf(maxDefocusResponse, minCorrespondenceResponse,
		defocusConfidence, correspondenceConfidence);

	// translate label map into depth map
	oclMat alphaMap, confidence, extendedDepthOfFieldImage;

	mask1 = (labels == 0);
	defocusAlpha.copyTo(alphaMap, mask1);
	defocusConfidence.copyTo(confidence, mask1);
	dEDOF.copyTo(extendedDepthOfFieldImage, mask1);

	mask1 = (labels == 1);
	correspondenceAlpha.copyTo(alphaMap, mask1);
	correspondenceConfidence.copyTo(confidence, mask1);
	cEDOF.copyTo(extendedDepthOfFieldImage, mask1);


	/*
	oclMat depth = pickDepthWithMaxConfidence(maxDefocusResponse,
		minCorrespondenceResponse, defocusConfidence, correspondenceConfidence);
	*/

	// 4) compute actual depth from alpha values
	oclMat depthMap;
	ocl::multiply(lightfield.getRawFocalLength(), alphaMap, depthMap);
	//oclMat depthMap = (lightfield.getRawFocalLength() * alphaMap);
	// TODO

	// TODO/debug save to attributes
	//renderer->setFocalLength(?);
	oclMat image = renderer->renderImage();

	//normalize(maxDefocusResponse, maxDefocusResponse, 0, 1, NORM_MINMAX);
	//normalize(minCorrespondenceResponse, minCorrespondenceResponse, 0, 1, NORM_MINMAX);
	//normalize(depth, depth, 0, 1, NORM_MINMAX);
	//normalize(image, image, 0, 1, NORM_MINMAX);

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
	normalize(m,m,0,1,NORM_MINMAX);
	string window3 = "confidence from defocus";
	namedWindow(window3, WINDOW_NORMAL);
	imshow(window3, m);
	
	correspondenceConfidence.download(m);
	normalize(m, m, 0, 1, NORM_MINMAX);
	string window4 = "confidence from correspondence";
	namedWindow(window4, WINDOW_NORMAL);
	imshow(window4, m);
	
	image.download(m);
	normalize(m, m, 0, 1, NORM_MINMAX);
	string window5 = "central perspective";
	namedWindow(window5, WINDOW_NORMAL);
	imshow(window5, m);
	/*
	alphaMap.download(m); normalize(m,m,0,1,NORM_MINMAX);
	normalize(m, m, 0, 1, NORM_MINMAX);
	string window6 = "combined depth map";
	namedWindow(window6, WINDOW_NORMAL);
	imshow(window6, m);
	*/
	waitKey(0);

	return alphaMap;
}


oclMat CDCDepthEstimator::calculateDefocusResponse(
	const LightFieldPicture& lightfield, const oclMat& refocusedImage,
	const float alpha)
{
	oclMat response, d2x, d2y;

	//ocl::Laplacian(refocusedImage, response, CV_32F, LAPLACIAN_KERNEL_SIZE); // Größe wird nicht unterstützt
	
	//ocl::Scharr(refocusedImage, d2x, CV_32FC1, 1, 0, LAPLACIAN_KERNEL_SIZE);
	//ocl::Scharr(refocusedImage, d2y, CV_32FC1, 0, 1, LAPLACIAN_KERNEL_SIZE);
	//ocl::Sobel(refocusedImage, d2x, CV_32FC1, 2, 0, LAPLACIAN_KERNEL_SIZE);
	//ocl::Sobel(refocusedImage, d2y, CV_32FC1, 0, 2, LAPLACIAN_KERNEL_SIZE);
	//response = d2x + d2y;
	
	ocl::filter2D(refocusedImage, response, DDEPTH, LoG,
		WINDOW_CENTER, 0, BORDER_TYPE);

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
			variance += squaredDifference;
		}
	variance /= Nuv;

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

	confidence1 *= 1. / maxVal;
	confidence2 *= 1. / maxVal;
	//ocl::divide(maxVal, confidence1, confidence1);
	//ocl::divide(maxVal, confidence2, confidence2);
}


// oclMar besitzt keinen Elementzugriff. Daher Mat verwenden oder eigenen Kernel entwickeln.
Mat CDCDepthEstimator::pickDepthWithMaxConfidence(Mat& depth1,
	Mat& depth2, Mat& confidence1, Mat& confidence2)
{
	Mat depth = Mat(depth1.size(), CV_32FC1);
	typedef float elementType;
	MatIterator_<elementType> itd0, itd1, itd2, itc1, itc2, end1;
	for (
		itd0 = depth.begin<elementType>(),
		itd1 = depth1.begin<elementType>(),
		itd2 = depth2.begin<elementType>(),
		itc1 = confidence1.begin<elementType>(),
		itc2 = confidence2.begin<elementType>(),
		end1 = depth1.end<elementType>();
		itd1 != end1; ++itd0, ++itd1, ++itd2, ++itc1, ++itc2 )
	{
		*itd0 = (*itc1 > *itc2) ? *itd1 : *itd2;
	}

	return depth;
}


MRF::CostVal CDCDepthEstimator::fnCost(int pix1, int pix2, MRF::Label i, MRF::Label j)
{
	return 1;
}


oclMat CDCDepthEstimator::mrf(const oclMat& depth1, const oclMat& depth2,
	const oclMat& confidence1, const oclMat& confidence2)
{
	MRF* mrf;
	EnergyFunction *energy;
	float time;
	
	// pre-calculate cost
	vector<float> dataCost1, dataCost2;
	Mat tmpMat;
	oclMat aDiffs, gradientX, gradientY, laplacian, dataCost, flatnessCost, smoothnessCost, totalCost;
	ocl::absdiff(depth1, depth2, aDiffs);

	// calculate cost for defocus solution
	ocl::multiply(aDiffs, confidence2, dataCost);
	dataCost *= LAMBDA_SOURCE[0];

	ocl::Sobel(depth1, gradientX, -1, 1, 0);
	ocl::Sobel(depth1, gradientY, -1, 0, 1);
	ocl::abs(gradientX, gradientX);
	ocl::abs(gradientY, gradientY);
	flatnessCost = gradientX + gradientY;

	ocl::Laplacian(depth1, laplacian, CV_32F);
	ocl::abs(laplacian, laplacian);
	smoothnessCost = laplacian * LAMBDA_SMOOTH;

	totalCost = dataCost + flatnessCost + smoothnessCost;
	totalCost.download(tmpMat);
	tmpMat.reshape(0, 1).copyTo(dataCost1);

	// calculate cost for corresponence solution
	ocl::multiply(aDiffs, confidence1, dataCost);
	dataCost *= LAMBDA_SOURCE[1];

	ocl::Sobel(depth2, gradientX, -1, 1, 0);
	ocl::Sobel(depth2, gradientY, -1, 0, 1);
	ocl::abs(gradientX, gradientX);
	ocl::abs(gradientY, gradientY);
	flatnessCost = gradientX + gradientY;

	ocl::Laplacian(depth2, laplacian, CV_32F);
	ocl::abs(laplacian, laplacian);
	smoothnessCost = laplacian * LAMBDA_SMOOTH;

	totalCost = dataCost + flatnessCost + smoothnessCost;
	totalCost.download(tmpMat);
	tmpMat.reshape(0, 1).copyTo(dataCost2);

	dataCost1.insert(dataCost1.end(), dataCost2.begin(), dataCost2.end());

	// define/generate complete cost function
	DataCost *data         = new DataCost(&dataCost1[0]);
	SmoothnessCost *smooth = new SmoothnessCost(&CDCDepthEstimator::fnCost);
	energy = new EnergyFunction(data,smooth);

	// compute optimized depth map (labeling)
	mrf = new MaxProdBP(depth1.size().width, depth1.size().height, 2, energy);
	mrf->initialize();
	mrf->clearAnswer();
	
	/*
	// debugging
	printf("Energy at the Start = %g (%g + %g)\n", (float)mrf->totalEnergy(),
	   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());
	*/
	// perform optimization
	oclMat newLabels, tmpOclMat;
	oclMat oldLabels = oclMat(depth1.size(), CV_32FC1, Scalar(2));
	double rootMeanSquareDeviation;
	int pixelCount = depth1.size().area();
	do {
		// perform more optimization
		mrf->optimize(1, time);	// TODO use constant

		// calculate root-mean-square deviation
		newLabels = oclMat(depth1.size(), CV_8UC1, mrf->getAnswerPtr());
		newLabels.convertTo(newLabels, CV_32F);
		tmpOclMat = newLabels - oldLabels;
		ocl::multiply(tmpOclMat, tmpOclMat, tmpOclMat);
		rootMeanSquareDeviation = std::sqrt(ocl::sum(tmpOclMat)[0] / (double) pixelCount);
		
		newLabels.copyTo(oldLabels);
		
		/*
		// debugging
		printf("Current energy = %g (%g + %g)\n", (float)mrf->totalEnergy(),
			(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());
		*/
	} while (rootMeanSquareDeviation > CONVERGENCE_FRACTION);

	/*
	// debugging
	string window7 = "labels";
	namedWindow(window7, WINDOW_NORMAL);
	imshow(window7, newLabels);
	*/

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
