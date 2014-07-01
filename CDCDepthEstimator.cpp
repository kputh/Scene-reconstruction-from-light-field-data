#include <iostream>	// debug
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>	// debug
#include "ImageRenderer1.h"
#include "CDCDepthEstimator.h"


const float CDCDepthEstimator::ALPHA_MIN					= 0.2;
const float CDCDepthEstimator::ALPHA_MAX					= 2.0;
const float CDCDepthEstimator::ALPHA_STEP					= 0.1;//0.007;
const Size CDCDepthEstimator::DEFOCUS_WINDOW_SIZE			= Size(9, 9);
const Size CDCDepthEstimator::CORRESPONDENCE_WINDOW_SIZE	= Size(9, 9);

const int CDCDepthEstimator::DDEPTH					= -1;
const Point CDCDepthEstimator::WINDOW_CENTER		= Point (-1, -1);
const int CDCDepthEstimator::BORDER_TYPE			= BORDER_DEFAULT; // TODO 
const Mat CDCDepthEstimator::DEFOCUS_WINDOW			= Mat(DEFOCUS_WINDOW_SIZE,
	CV_32F, Scalar(1.0 / DEFOCUS_WINDOW_SIZE.area()));
const Mat CDCDepthEstimator::CORRESPONDENCE_WINDOW	= Mat(DEFOCUS_WINDOW_SIZE,
	CV_32F, Scalar(1.0 / CORRESPONDENCE_WINDOW_SIZE.area()));


CDCDepthEstimator::CDCDepthEstimator(void)
{
	this->renderer =  new ImageRenderer1;
}


CDCDepthEstimator::~CDCDepthEstimator(void)
{
}


Mat CDCDepthEstimator::estimateDepth(const LightFieldPicture lightfield)
{
	cout << "Schritt 1" << endl;
	// initialize Da, Ca
	vector<Mat> responses;

	// 1) for each shear, compute depth response
	this->renderer->setLightfield(lightfield);
	const int stepCount = floor((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP);
	Mat response;
	for (float alpha = ALPHA_MIN; alpha <= ALPHA_MAX; alpha += ALPHA_STEP)
	{
		response = Mat(lightfield.SPARTIAL_RESOLUTION, CV_32FC3,
			Scalar(0, 0, alpha));

		calculateDefocusResponse(lightfield, response, alpha);
		calculateCorrespondenceResponse(lightfield, response, alpha);

		responses.push_back(response);
		// TODO eine dreidimensionale Matrix verwenden
	}

	cout << "Schritt 2" << endl;
	// 2) for each pixel, compute response optimum
	Mat maxDefocusResponses = argMaxAlpha(responses);
	Mat minCorrespondenceResponses = argMinAlpha(responses);

	Mat defocusConfidence = calculateConfidence(maxDefocusResponses);
	Mat correspondenceConfidence = calculateConfidence(minCorrespondenceResponses);

	// reduce to first extremum
	Mat maxDefocusResponse = getFirstExtremum(maxDefocusResponses);
	Mat minCorrespondenceResponse = getFirstExtremum(minCorrespondenceResponses);

	// 3) global operation to combine cues
	/*
	Mat depth = mrf(maxDefocusResponse, minCorrespondenceResponse,
		defocusConfidence, correspondenceConfidence);
	*/
	Mat depth = pickDepthWithMaxConfidence(maxDefocusResponse,
		minCorrespondenceResponse, defocusConfidence, correspondenceConfidence);

	// 4) compute actual depth from focal length
	// TODO

	// TODO/debug save to attributes
	//renderer->setFocalLength(?);
	Mat image = renderer->renderImage();
	// TODO Werte sind außerhalb des erwarteten Bereichs DEBUGGEN!!!
	normalize(maxDefocusResponse, maxDefocusResponse, 0, 1, NORM_MINMAX);
	normalize(minCorrespondenceResponse, minCorrespondenceResponse, 0, 1, NORM_MINMAX);
	normalize(defocusConfidence, defocusConfidence, 0, 1, NORM_MINMAX);
	normalize(correspondenceConfidence, correspondenceConfidence, 0, 1, NORM_MINMAX);
	normalize(depth, depth, 0, 1, NORM_MINMAX);
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
	Mat& response, float alpha)
{
	double newFocalLength = alpha * lightfield.getRawFocalLength();
	this->renderer->setFocalLength(newFocalLength);

	vector<Mat> channels;
	split(response, channels);
	const int i = 0;

	channels[i] = this->renderer->renderImage();
	cvtColor(channels[i], channels[i], CV_RGB2GRAY);	// TODO anders lösen
	Laplacian(channels[i], channels[i], CV_32F, DEFOCUS_WINDOW.size().width, 1,
		0, BORDER_TYPE);
	channels[i] = abs(channels[i]);
	filter2D(channels[i], channels[i], DDEPTH, DEFOCUS_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);

	merge(channels, response);
}


void CDCDepthEstimator::calculateCorrespondenceResponse(
	LightFieldPicture lightfield, Mat& response, float alpha)
{
	double newFocalLength = alpha * lightfield.getRawFocalLength();
	this->renderer->setFocalLength(newFocalLength);

	vector<Mat> channels;
	split(response, channels);
	const int i = 1;

	channels[i] = this->renderer->renderImage();
	cvtColor(channels[i], channels[i], CV_RGB2GRAY);	// TODO anders lösen
	Mat subapertureImage, differenceImage, squaredDifference;
	Mat variance = Mat::zeros(lightfield.SPARTIAL_RESOLUTION, CV_32FC1);

	int u, v;
	for (v = 0; v < lightfield.ANGULAR_RESOLUTION.height; v++)
	{
		for (u = 0; u < lightfield.ANGULAR_RESOLUTION.width; u++)
		{
			subapertureImage = lightfield.getSubapertureImage(u, v);
			cvtColor(subapertureImage, subapertureImage, CV_RGB2GRAY);	// TODO anders lösen
			differenceImage = subapertureImage - channels[i];
			multiply(differenceImage, differenceImage, squaredDifference);
			variance += squaredDifference;
		}
	}
	variance /= lightfield.ANGULAR_RESOLUTION.area();

	Mat standardDeviation, confidence;
	sqrt(variance, standardDeviation);
	filter2D(standardDeviation, confidence, DDEPTH, CORRESPONDENCE_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	
	channels[i] = confidence;
	merge(channels, response);
}


Mat CDCDepthEstimator::argMaxAlpha(vector<Mat> responses)
{
	// sort responses until the two greatest values have been found
	typedef Vec3f elementType;
	const int i = 0;	// index of defocus response
	Mat mat1, mat2;
	int height = responses[0].size().height;
	int width = responses[0].size().width;
	elementType tmp;
	float response1, response2;
	MatIterator_<elementType> it1, it2, end1;
	int loopIndex, responseIndex;
	for (loopIndex = 0; loopIndex < 2; loopIndex++)
	{
		for (responseIndex = 1; responseIndex < responses.size() - loopIndex; responseIndex++)
		{
			mat1 = responses[responseIndex - 1];
			mat2 = responses[responseIndex];

            for( it1 = mat1.begin<elementType>(), it2 = mat2.begin<elementType>(),
				end1 = mat1.end<elementType>(); it1 != end1; ++it1, ++it2 )
            {
				response1	= (*it1)[i];
				response2	= (*it2)[i];

				if (response1 > response2)	// compare responses
				{
					// swap responses with attached alpha values
					tmp = (*it1);
					(*it1) = (*it2);
					(*it2) = tmp;
				}
            }
		}
	}

	// reduce to alpha values
	int firstExtremumIndex		= responses.size() - 1;
	int secondExtremumIndex		= responses.size() - 2;
	Mat alphas = Mat(responses[0].size(), CV_32FC2);
	Mat in[]		= { responses[firstExtremumIndex],
		responses[secondExtremumIndex] };
	Mat out[]		= { alphas };
	int from_to[]	= { 2,0, 5,1 };
	mixChannels(in, 2, out, 1, from_to, 2);

	return alphas;
}


Mat CDCDepthEstimator::argMinAlpha(vector<Mat> responses)
{
	// sort responses until the two smallest values have been found
	typedef Vec3f elementType;
	const int i = 1;	// index of corresponcence response
	Mat mat1, mat2;
	int height = responses[0].size().height;
	int width = responses[0].size().width;
	elementType tmp;
	float response1, response2;
	MatIterator_<elementType> it1, it2, end1;
	int loopIndex, responseIndex;
	for (loopIndex = 0; loopIndex < 2; loopIndex++)
	{
		for (responseIndex = 1; responseIndex < responses.size() - loopIndex; responseIndex++)
		{
			mat1 = responses[responseIndex - 1];
			mat2 = responses[responseIndex];

            for( it1 = mat1.begin<elementType>(), it2 = mat2.begin<elementType>(),
				end1 = mat1.end<elementType>(); it1 != end1; ++it1, ++it2 )
            {
				response1	= (*it1)[i];
				response2	= (*it2)[i];

				if (response1 < response2)	// compare responses
				{
					// swap responses with attached alpha values
					tmp = (*it1);
					(*it1) = (*it2);
					(*it2) = tmp;
				}
            }
		}
	}

	// reduce to alpha values
	int firstExtremumIndex		= responses.size() - 1;
	int secondExtremumIndex		= responses.size() - 2;
	Mat alphas = Mat(responses[0].size(), CV_32FC2);
	Mat in[]		= { responses[firstExtremumIndex],
		responses[secondExtremumIndex] };
	Mat out[]		= { alphas };
	int from_to[]	= { 2,0, 5,1 };
	mixChannels(in, 2, out, 1, from_to, 2);

	return alphas;
}


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


Mat CDCDepthEstimator::mrf(Mat depth1, Mat depth2, Mat confidence1, Mat confidence2)
{
	// TODO ...

	Mat depth = Mat(depth1.size(), CV_32FC1);
	return depth;
}
