#include <iostream>	// debugging
#include <cfloat>	// debugging
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>	// debugging
#include "mrf.h"
#include "MaxProdBP.h"
#include "ImageRenderer1.h"
#include "CDCDepthEstimator.h"


const float CDCDepthEstimator::ALPHA_MIN					= 0.2;
const float CDCDepthEstimator::ALPHA_MAX					= 2.0;
const int CDCDepthEstimator::DEPTH_RESOLUTION				= 100;
const Size CDCDepthEstimator::DEFOCUS_WINDOW_SIZE			= Size(9, 9);
const Size CDCDepthEstimator::CORRESPONDENCE_WINDOW_SIZE	= Size(9, 9);
const float CDCDepthEstimator::LAMBDA_SOURCE[]				= { 1, 1 };
const float CDCDepthEstimator::LAMBDA_FLAT					= 2;
const float CDCDepthEstimator::LAMBDA_SMOOTH				= 2;
const double CDCDepthEstimator::CONVERGENCE_FRACTION		= 1;

const int CDCDepthEstimator::DDEPTH					= -1;
const Point CDCDepthEstimator::WINDOW_CENTER		= Point (-1, -1);
const int CDCDepthEstimator::BORDER_TYPE			= BORDER_DEFAULT; // TODO 
const Mat CDCDepthEstimator::DEFOCUS_WINDOW
	= Mat(DEFOCUS_WINDOW_SIZE, CV_32F,
	Scalar(1 / (float) DEFOCUS_WINDOW_SIZE.area()));
const Mat CDCDepthEstimator::CORRESPONDENCE_WINDOW
	= Mat(CORRESPONDENCE_WINDOW_SIZE, CV_32F,
	Scalar(1 / (float) CORRESPONDENCE_WINDOW_SIZE.area()));


CDCDepthEstimator::CDCDepthEstimator(void)
{
	this->renderer =  new ImageRenderer1;
}


CDCDepthEstimator::~CDCDepthEstimator(void)
{
}


oclMat CDCDepthEstimator::estimateDepth(const LightFieldPicture& lightfield)
{
	cout << "Schritt 1" << endl;
	// initialize Da, Ca
	vector<oclMat> responses;

	// 1) for each shear, compute depth response
	this->renderer->setLightfield(lightfield);
	const float alphaStep = (ALPHA_MAX - ALPHA_MIN) / (float) DEPTH_RESOLUTION;
	oclMat refocusedImage, response;
	for (float alpha = ALPHA_MIN; alpha <= ALPHA_MAX; alpha += alphaStep)
	{
		this->renderer->setAlpha(alpha);
		refocusedImage = this->renderer->renderImage();
		ocl::cvtColor(refocusedImage, refocusedImage, CV_RGB2GRAY);	// TODO anders lösen

		response = oclMat(lightfield.SPARTIAL_RESOLUTION, CV_32FC3,
			Scalar(0, 0, alpha));

		calculateDefocusResponse(lightfield, response, alpha, refocusedImage);
		calculateCorrespondenceResponse(lightfield, response, alpha, refocusedImage);

		responses.push_back(response);
	}

	cout << "Schritt 2" << endl;
	// 2) for each pixel, compute response optimum
	// find maximum defocus response per pixel
	oclMat maxDefocusResponses			= argMaxAlpha(responses);
	// find minimum corresponcence response per pixel
	oclMat minCorrespondenceResponses	= argMinAlpha(responses);

	oclMat defocusConfidence			= calculateConfidence(maxDefocusResponses);
	oclMat correspondenceConfidence		= calculateConfidence(minCorrespondenceResponses);

	// reduce to first extremum
	oclMat maxDefocusResponse			= getFirstExtremum(maxDefocusResponses);
	oclMat minCorrespondenceResponse	= getFirstExtremum(minCorrespondenceResponses);

	// normalize confidence
	normalizeConfidence(defocusConfidence, correspondenceConfidence);

	cout << "Schritt 3" << endl;
	// 3) global operation to combine cues
	oclMat depth = mrf(maxDefocusResponse, minCorrespondenceResponse,
		defocusConfidence, correspondenceConfidence);
	/*
	oclMat depth = pickDepthWithMaxConfidence(maxDefocusResponse,
		minCorrespondenceResponse, defocusConfidence, correspondenceConfidence);
	*/

	// 4) compute actual depth from focal length
	// TODO

	// TODO/debug save to attributes
	//renderer->setFocalLength(?);
	oclMat image = renderer->renderImage();

	//normalize(maxDefocusResponse, maxDefocusResponse, 0, 1, NORM_MINMAX);
	//normalize(minCorrespondenceResponse, minCorrespondenceResponse, 0, 1, NORM_MINMAX);
	//normalize(depth, depth, 0, 1, NORM_MINMAX);
	//normalize(image, image, 0, 1, NORM_MINMAX);

	string window1 = "depth from defocus";
	namedWindow(window1, WINDOW_NORMAL);
	imshow(window1, maxDefocusResponse);

	string window2 = "depth from correspondence";
	namedWindow(window2, WINDOW_NORMAL);
	imshow(window2, minCorrespondenceResponse);

	string window3 = "confidence from defocus";
	namedWindow(window3, WINDOW_NORMAL);
	imshow(window3, defocusConfidence);

	string window4 = "confidence from correspondence";
	namedWindow(window4, WINDOW_NORMAL);
	imshow(window4, correspondenceConfidence);

	string window5 = "central perspective";
	namedWindow(window5, WINDOW_NORMAL);
	imshow(window5, image);

	string window6 = "combined depth map";
	namedWindow(window6, WINDOW_NORMAL);
	imshow(window6, depth);

	waitKey(0);

	return depth;
}


void CDCDepthEstimator::calculateDefocusResponse(const LightFieldPicture& lightfield,
	oclMat& responses, float alpha, const oclMat& refocusedImage)
{
	vector<oclMat> channels;
	ocl::split(responses, channels);
	const int i = 0;

	oclMat response;
	ocl::Laplacian(refocusedImage, response, CV_32F);
	ocl::abs(response, response);
	ocl::filter2D(response, channels[i], DDEPTH, DEFOCUS_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);

	ocl::merge(channels, responses);
}


void CDCDepthEstimator::calculateCorrespondenceResponse(
	const LightFieldPicture& lightfield, oclMat& response, float alpha, const oclMat& refocusedImage)
{
	vector<oclMat> channels;
	ocl::split(response, channels);
	const int i = 1;

	oclMat subapertureImage, differenceImage, squaredDifference;
	oclMat variance = oclMat(lightfield.SPARTIAL_RESOLUTION, CV_32FC1, Scalar(0));

	int u, v;
	for (v = 0; v < lightfield.ANGULAR_RESOLUTION.height; v++)
		for (u = 0; u < lightfield.ANGULAR_RESOLUTION.width; u++)
		{
			subapertureImage = lightfield.getSubapertureImageI(u, v); // TODO reelle Koordinaten verwenden
			ocl::cvtColor(subapertureImage, subapertureImage, CV_RGB2GRAY);	// TODO anders lösen
			ocl::subtract(subapertureImage, refocusedImage, differenceImage);	// FEHLER sub-aperture image muss verschoben sein
			ocl::multiply(differenceImage, differenceImage, squaredDifference);
			ocl::add(variance, squaredDifference, variance);
		}
	ocl::divide(lightfield.ANGULAR_RESOLUTION.area(), variance, variance);

	oclMat standardDeviation, confidence;
	ocl::pow(variance, 0.5, standardDeviation);	// es gibt kein ocl::sqrt()
	ocl::filter2D(standardDeviation, confidence, DDEPTH, CORRESPONDENCE_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	
	channels[i] = confidence;
	ocl::merge(channels, response);
}


oclMat CDCDepthEstimator::argMaxAlpha(const vector<oclMat>& responses) const
{
	typedef Vec3f elementType;	// (defocus response, correspondence response, alpha)
	const int responseIndex = 0;	// index of defocus response

	int height			= responses[0].size().height;
	int width			= responses[0].size().width;
	int responseCount	= responses.size();

	oclMat matrix, max1, max2;
	const oclMat multiplier = oclMat(responses[0].size(), responses[0].type(),
		Scalar(1000, 1, 1, 1));
	responses[0].copyTo(max1);
	responses[0].copyTo(max2);
	ocl::multiply(multiplier, max1, max1);
	ocl::multiply(multiplier, max2, max2);

	for (int responseIndex = 1; responseIndex < responseCount; responseIndex++)
	{
		ocl::multiply(responses[responseIndex], multiplier, matrix);

		// find first maximum
		ocl::max(matrix, max1, max1);

		// find second maximum
		ocl::min(matrix, max1, matrix);
		ocl::max(matrix, max2, max2);
	}

	// reduce to alpha values
	vector<oclMat> channels1, channels2, resultChannels(2);

	ocl::split(max1, channels1);
	ocl::split(max2, channels2);

	resultChannels[0] = channels1[2];
	resultChannels[1] = channels2[2];

	oclMat alphas;
	ocl::merge(resultChannels, alphas);

	return alphas;
}


oclMat CDCDepthEstimator::argMinAlpha(const vector<oclMat>& responses) const
{
	typedef Vec3f elementType;	// (defocus response, correspondence response, alpha)
	const int responseIndex = 1;	// index of correspondence response

	int height			= responses[0].size().height;
	int width			= responses[0].size().width;
	int responseCount	= responses.size();

	oclMat matrix, min1, min2;
	const oclMat multiplier = oclMat(responses[0].size(), responses[0].type(),
		Scalar(1, 1000, 1, 1));
	responses[0].copyTo(min1);
	responses[0].copyTo(min2);
	ocl::multiply(multiplier, min1, min1);
	ocl::multiply(multiplier, min2, min2);

	for (int responseIndex = 1; responseIndex < responseCount; responseIndex++)
	{
		ocl::multiply(responses[responseIndex], multiplier, matrix);

		// find first maximum
		ocl::min(matrix, min1, min1);

		// find second maximum
		ocl::max(matrix, min1, matrix);
		ocl::min(matrix, min2, min2);
	}

	// reduce to alpha values
	vector<oclMat> channels1, channels2, resultChannels(2);

	ocl::split(min1, channels1);
	ocl::split(min2, channels2);

	resultChannels[0] = channels1[2];
	resultChannels[1] = channels2[2];

	oclMat alphas;
	ocl::merge(resultChannels, alphas);

	return alphas;
}


// peak ratio confidence
oclMat CDCDepthEstimator::calculateConfidence(const oclMat& extrema)
{
	vector<oclMat> channels;
	oclMat confidence = oclMat(extrema.size(), CV_32FC1);

	ocl::split(extrema, channels);
	ocl::divide(channels[0], channels[1], confidence);

	return confidence;
}


oclMat CDCDepthEstimator::getFirstExtremum(const oclMat& extrema)
{
	vector<oclMat> channels;
	ocl::split(extrema, channels);

	return channels[0];
}


void CDCDepthEstimator::normalizeConfidence(oclMat& confidence1, oclMat& confidence2)
{
	// find greatest confidence in both matrices combined
	oclMat maxConfidenceMat;
	double minVal, maxVal;

	ocl::max(confidence1, confidence2, maxConfidenceMat);
	ocl::minMax(maxConfidenceMat, &minVal, &maxVal);

	ocl::divide(maxVal, confidence1, confidence1);
	ocl::divide(maxVal, confidence2, confidence2);
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
	ocl::add(gradientX, gradientY, flatnessCost);

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
	ocl::add(gradientX, gradientY, flatnessCost);

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
	// translate label map into depth map
	oclMat optimizedDepth;
	tmpOclMat = newLabels - 1;
	ocl::abs(tmpOclMat, tmpOclMat);
	ocl::multiply(depth1, tmpOclMat, optimizedDepth);
	ocl::multiply(depth2, newLabels, tmpOclMat);
	optimizedDepth += tmpOclMat;

	delete mrf;

	return optimizedDepth;
}
