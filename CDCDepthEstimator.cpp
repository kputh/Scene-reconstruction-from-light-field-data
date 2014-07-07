#include <iostream>	// debug
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>	// debug
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
const Mat CDCDepthEstimator::DEFOCUS_WINDOW			= Mat(DEFOCUS_WINDOW_SIZE,
	CV_32F, Scalar(1.0 / DEFOCUS_WINDOW_SIZE.area()));
const Mat CDCDepthEstimator::CORRESPONDENCE_WINDOW	= Mat(
	CORRESPONDENCE_WINDOW_SIZE, CV_32F,
	Scalar(1 / (float) CORRESPONDENCE_WINDOW_SIZE.area()));

vector<float> CDCDepthEstimator::fsCost[2];


CDCDepthEstimator::CDCDepthEstimator(void)
{
	this->renderer =  new ImageRenderer1;
}


CDCDepthEstimator::~CDCDepthEstimator(void)
{
}


Mat CDCDepthEstimator::estimateDepth(LightFieldPicture lightfield)
{
	cout << "Schritt 1" << endl;
	// initialize Da, Ca
	vector<Mat> responses;

	// 1) for each shear, compute depth response
	this->renderer->setLightfield(lightfield);
	const float alphaStep = (ALPHA_MAX - ALPHA_MIN) / (float) DEPTH_RESOLUTION;
	Mat refocusedImage, response;
	double newFocalLength;
	for (float alpha = ALPHA_MIN; alpha <= ALPHA_MAX; alpha += alphaStep)
	{
		newFocalLength = alpha * lightfield.getRawFocalLength();
		this->renderer->setFocalLength(newFocalLength);	// TODO direkt alpha übergeben
		refocusedImage = this->renderer->renderImage();
		cvtColor(refocusedImage, refocusedImage, CV_RGB2GRAY);	// TODO anders lösen

		response = Mat(lightfield.SPARTIAL_RESOLUTION, CV_32FC3,
			Scalar(0, 0, alpha));

		calculateDefocusResponse(lightfield, response, alpha, refocusedImage);
		calculateCorrespondenceResponse(lightfield, response, alpha, refocusedImage);

		responses.push_back(response);
	}

	cout << "Schritt 2" << endl;
	// 2) for each pixel, compute response optimum
	// find maximum defocus response per pixel
	Mat maxDefocusResponses			= argExtrAlpha(responses,
		&CDCDepthEstimator::isGreaterThan, 0);
	// find minimum corresponcence response per pixel
	Mat minCorrespondenceResponses	= argExtrAlpha(responses,
		&CDCDepthEstimator::isLesserThan, 1);

	Mat defocusConfidence			= calculateConfidence(maxDefocusResponses);
	Mat correspondenceConfidence	= calculateConfidence(minCorrespondenceResponses);

	// reduce to first extremum
	Mat maxDefocusResponse = getFirstExtremum(maxDefocusResponses);
	Mat minCorrespondenceResponse = getFirstExtremum(minCorrespondenceResponses);

	// normalize confidence
	normalizeConfidence(defocusConfidence, correspondenceConfidence);

	cout << "Schritt 3" << endl;
	// 3) global operation to combine cues
	Mat depth = mrf(maxDefocusResponse, minCorrespondenceResponse,
		defocusConfidence, correspondenceConfidence);
	/*
	Mat depth = pickDepthWithMaxConfidence(maxDefocusResponse,
		minCorrespondenceResponse, defocusConfidence, correspondenceConfidence);
	*/

	// 4) compute actual depth from focal length
	// TODO

	// TODO/debug save to attributes
	//renderer->setFocalLength(?);
	Mat image = renderer->renderImage();
	// TODO Werte sind außerhalb des erwarteten Bereichs DEBUGGEN!!!

	//normalize(maxDefocusResponse, maxDefocusResponse, 0, 1, NORM_MINMAX);
	//normalize(minCorrespondenceResponse, minCorrespondenceResponse, 0, 1, NORM_MINMAX);
	//normalize(depth, depth, 0, 1, NORM_MINMAX);
	normalize(image, image, 0, 1, NORM_MINMAX);

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


void CDCDepthEstimator::calculateDefocusResponse(LightFieldPicture lightfield,
	Mat& response, float alpha, Mat refocusedImage)
{
	vector<Mat> channels;
	split(response, channels);
	const int i = 0;

	Laplacian(refocusedImage, refocusedImage, CV_32F, DEFOCUS_WINDOW_SIZE.width, 1,
		0, BORDER_TYPE);
	filter2D(abs(refocusedImage), channels[i], DDEPTH, DEFOCUS_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);

	merge(channels, response);
}


void CDCDepthEstimator::calculateCorrespondenceResponse(
	LightFieldPicture lightfield, Mat& response, float alpha, Mat refocusedImage)
{
	vector<Mat> channels;
	split(response, channels);
	const int i = 1;

	Mat subapertureImage, differenceImage, squaredDifference;
	Mat variance = Mat::zeros(lightfield.SPARTIAL_RESOLUTION, CV_32FC1);

	int u, v;
	for (v = 0; v < lightfield.ANGULAR_RESOLUTION.height; v++)
		for (u = 0; u < lightfield.ANGULAR_RESOLUTION.width; u++)
		{
			subapertureImage = lightfield.getSubapertureImage(u, v); // FEHLER (?) das Lichtfeld muss refokussiert sein
			cvtColor(subapertureImage, subapertureImage, CV_RGB2GRAY);	// TODO anders lösen
			differenceImage = subapertureImage - refocusedImage;
			multiply(differenceImage, differenceImage, squaredDifference);
			variance += squaredDifference;
		}
	variance /= lightfield.ANGULAR_RESOLUTION.area();

	Mat standardDeviation, confidence;
	sqrt(variance, standardDeviation);
	filter2D(standardDeviation, confidence, DDEPTH, CORRESPONDENCE_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	
	channels[i] = confidence;
	merge(channels, response);
}


bool CDCDepthEstimator::isGreaterThan(float a, float b)
{
	return a > b;
}


bool CDCDepthEstimator::isLesserThan(float a, float b)
{
	return a < b;
}


Mat CDCDepthEstimator::argExtrAlpha(vector<Mat> responses, bool (*isExtr)(float,float),
	const int responseChannelIndex)
{
	typedef Vec3f elementType;	// (defocus response, correspondence response, alpha)

	int height			= responses[0].size().height;
	int width			= responses[0].size().width;
	int responseCount	= responses.size();

	Mat max1, max2;
	responses[0].copyTo(max1);
	responses[0].copyTo(max2);

	float response;
	int x, y, responseIndex;
	for (y = 0; y < height; y++)
		for (x = 0; x < width; x++)
			for (responseIndex = 1; responseIndex < responseCount; responseIndex++)
			{
				response = responses[responseIndex].at<elementType>(y, x)
					[responseChannelIndex];

				if (isExtr(response,
						max1.at<elementType>(y, x)[responseChannelIndex]))
					max1.at<elementType>(y, x) = responses[responseIndex].at<elementType>(y, x);
				else if (isExtr(response,
						max2.at<elementType>(y, x)[responseChannelIndex]))
					max2.at<elementType>(y, x) = responses[responseIndex].at<elementType>(y, x);
			}

	// reduce to alpha values
	Mat alphas = Mat(responses[0].size(), CV_32FC2);
	Mat in[]		= { max1, max2 };
	Mat out[]		= { alphas };
	int from_to[]	= { 2,0, 5,1 };
	mixChannels(in, 2, out, 1, from_to, 2);

	return alphas;
}


// peak ratio confidence
Mat CDCDepthEstimator::calculateConfidence(Mat extrema)
{
	vector<Mat> channels;
	Mat confidence = Mat(extrema.size(), CV_32FC1);

	split(extrema, channels);
	divide(channels[0], channels[1], confidence);

	return confidence;
}


Mat CDCDepthEstimator::getFirstExtremum(Mat extrema)
{
	vector<Mat> channels;
	split(extrema, channels);

	return channels[0];
}


void CDCDepthEstimator::normalizeConfidence(Mat& confidence1, Mat& confidence2)
{
	// find greatest confidence in both matrices combined
	float maxConfidenceValue = -1;
	MatIterator_<float> it, end;
    for(it = confidence1.begin<float>(), end = confidence1.end<float>();
			it != end; ++it)
		if (*it > maxConfidenceValue)
			maxConfidenceValue = *it;
    for(it = confidence2.begin<float>(), end = confidence2.end<float>();
			it != end; ++it)
		if (*it > maxConfidenceValue)
			maxConfidenceValue = *it;

	confidence1 /= maxConfidenceValue;
	confidence2 /= maxConfidenceValue;
}


Mat CDCDepthEstimator::pickDepthWithMaxConfidence(Mat depth1, Mat depth2, Mat confidence1, Mat confidence2)
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
	return fsCost[i][pix2];
}

Mat CDCDepthEstimator::mrf(Mat depth1, Mat depth2, Mat confidence1, Mat confidence2)
{
	MRF* mrf;
	EnergyFunction *energy;
	float time;
	
	// pre-calculate data cost
	Mat aDiffs, weighed1, weighed2;
	absdiff(depth1, depth2, aDiffs);
	multiply(aDiffs, confidence2, weighed1);
	multiply(aDiffs, confidence1, weighed2);
	weighed1 *= LAMBDA_SOURCE[0];
	weighed2 *= LAMBDA_SOURCE[1];

	// save data cost in a single array
	vector<float> dataCost1, dataCost2;
	weighed1.reshape(0, 1).copyTo(dataCost1);
	weighed2.reshape(0, 1).copyTo(dataCost2);
	dataCost1.insert(dataCost1.end(), dataCost2.begin(), dataCost2.end());

	// pre-calculate flatness + smoothness cost
	Mat gradientX, gradientY, laplacian, flatnessCost, smoothnessCost, totalFSCost;
	Sobel(depth1, gradientX, -1, 1, 0);
	Sobel(depth1, gradientY, -1, 0, 1);
	add(abs(gradientX), abs(gradientY), flatnessCost);
	Laplacian(depth1, laplacian, CV_32F);
	smoothnessCost = abs(laplacian) * LAMBDA_SMOOTH;
	totalFSCost = smoothnessCost + flatnessCost;
	totalFSCost.reshape(0, 1).copyTo(fsCost[0]);

	Sobel(depth2, gradientX, -1, 1, 0);
	Sobel(depth2, gradientY, -1, 0, 1);
	add(abs(gradientX), abs(gradientY), flatnessCost);
	Laplacian(depth1, laplacian, CV_32F);
	smoothnessCost = abs(laplacian) * LAMBDA_SMOOTH;
	totalFSCost = smoothnessCost + flatnessCost;
	totalFSCost.reshape(0, 1).copyTo(fsCost[1]);

	// define/generate complete cost function
	DataCost *data         = new DataCost(&dataCost1[0]);
	SmoothnessCost *smooth = new SmoothnessCost(&CDCDepthEstimator::fnCost);
	energy = new EnergyFunction(data,smooth);

	// compute optimized depth map (labeling)
	mrf = new MaxProdBP(depth1.size().width, depth1.size().height, 2, energy);
	mrf->initialize();
	mrf->clearAnswer();
	
	// perform optimization
	Mat newLabels, tmp;
	//Mat cLabels = Mat(depth1.size(), CV_8UC1, mrf->getAnswerPtr());
	//Mat currentLabels;
	Mat oldLabels = Mat(depth1.size(), CV_32FC1, Scalar(2));
	double rootMeanSquareDeviation;
	int pixelCount = depth1.size().area();
	do {
	//for (int i = 0; i < 100; i++) {
		// perform more optimization
		mrf->optimize(1, time);	// TODO use constant

		// calculate root-mean-square deviation
		newLabels = Mat(depth1.size(), CV_8UC1, mrf->getAnswerPtr()); // kann die gleiche Mat kann in jedem Durchgang benutzt werden?
		newLabels.convertTo(newLabels, CV_32F);
		//cLabels.convertTo(currentLabels, CV_32F);
		tmp = newLabels - oldLabels;
		//tmp = currentLabels - oldLabels;
		multiply(tmp, tmp, tmp);
		rootMeanSquareDeviation = std::sqrt(sum(tmp)[0] / (double) pixelCount);
		
		// debugging
		cout << "sum(abs(newLabels - oldLabels))[0] = " << sum(abs(newLabels - oldLabels))[0] << " (0 indicates identity)" << endl;
		cout << "sum(newLabels)[0] = " << sum(newLabels)[0] << endl;

		newLabels.copyTo(oldLabels);
		//currentLabels.copyTo(oldLabels);
		
		// debugging
		cout << "rootMeanSquareDeviation = " << rootMeanSquareDeviation << endl;
		MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
		MRF::EnergyVal E_data   = mrf->dataEnergy();
		printf("Total Energy = %d (Smoothness energy %d, Data Energy %d)\n", E_smooth+E_data,E_smooth,E_data);

	} while (rootMeanSquareDeviation > CONVERGENCE_FRACTION);

	// translate label map into depth map
	Mat optimizedDepth = Mat::zeros(depth1.size(), depth1.type());
	accumulateProduct(depth1, abs(newLabels - 1), optimizedDepth);
	accumulateProduct(depth2, newLabels, optimizedDepth);

	delete mrf;

	return optimizedDepth;
}
