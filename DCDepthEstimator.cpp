#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageRenderer1.h"
#include "DCDepthEstimator.h"


const float CDCDepthEstimator::ALPHA_MIN					= 0.2;
const float CDCDepthEstimator::ALPHA_MAX					= 2.0;
const float CDCDepthEstimator::ALPHA_STEP					= 0.007;
const Size CDCDepthEstimator::DEFOCUS_WINDOW_SIZE			= Size(9, 9);
const Size CDCDepthEstimator::CORRESPONDENCE_WINDOW_SIZE	= Size(9, 9);

const int CDCDepthEstimator::DDEPTH					= -1;
const Point CDCDepthEstimator::WINDOW_CENTER		= Point (-1, -1);
const int CDCDepthEstimator::BORDER_TYPE			= BORDER_DEFAULT; // TODO 
const Mat CDCDepthEstimator::DEFOCUS_WINDOW			= Mat::ones(DEFOCUS_WINDOW_SIZE, CV_32F);
const Mat CDCDepthEstimator::CORRESPONDENCE_WINDOW	= Mat::ones(DEFOCUS_WINDOW_SIZE, CV_32F);


CDCDepthEstimator::CDCDepthEstimator(void)
{
}


CDCDepthEstimator::~CDCDepthEstimator(void)
{
}


Mat CDCDepthEstimator::estimateDepth(const LightFieldPicture lightfield)
{
	// initialize Da, Ca
	vector<Mat> defocusResponses;
	vector<Mat> correspondenceResponses;

	// 1) for each shear, compute depth response
	for (float alpha = ALPHA_MIN; alpha <= ALPHA_MAX; alpha += ALPHA_STEP)
	{
		defocusResponses.push_back(calculateDefocusResponse(lightfield, alpha));
		correspondenceResponses.push_back(calculateCorrespondenceResponse(lightfield, alpha));
	}

	// 2) for each pixel, compute response optimum
	Mat maxDefocusResponses = argMaxAlpha(defocusResponses);
	Mat minCorrespondenceResponses = argMinAlpha(correspondenceResponses);

	Mat defocusConfidence = calculateConfidence(maxDefocusResponses);
	Mat CorrespondenceConfidence = calculateConfidence(minCorrespondenceResponses);
	// TODO split channels for first and second extremum

	// 3) global operation to combine cues
	Mat depth = mrf(maxDefocusResponses, minCorrespondenceResponses,
		defocusConfidence, CorrespondenceConfidence);

	return depth;
}

Mat CDCDepthEstimator::calculateDefocusResponse(LightFieldPicture lightfield, float alpha)
{
	double newFocalLength = alpha * lightfield.getRawFocalLength();
	ImageRenderer1 renderer = ImageRenderer1();
	renderer.setLightfield(lightfield);	// einmal setzen
	renderer.setFocalLength(newFocalLength);

	Mat refocusedImage = renderer.renderImage();
	Mat laplacian;
	Laplacian(refocusedImage, laplacian, CV_32F); // Parameter anpassen
	Mat absLaplacian = abs(laplacian);
	Mat summedLaplacian;
	filter2D(absLaplacian, summedLaplacian, DDEPTH, DEFOCUS_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);

	return summedLaplacian;
}


Mat CDCDepthEstimator::calculateCorrespondenceResponse(LightFieldPicture lightfield, float alpha)
{
	double newFocalLength = alpha * lightfield.getRawFocalLength();
	ImageRenderer1 renderer = ImageRenderer1();
	renderer.setLightfield(lightfield);	// einmal setzen
	renderer.setFocalLength(newFocalLength);

	Mat refocusedImage = renderer.renderImage();
	Mat subapertureImage, differenceImage, squaredDifference;
	Mat variance = Mat::zeros(lightfield.SPARTIAL_RESOLUTION, CV_32FC1);

	int u, v;
	for (v = 0; v < lightfield.ANGULAR_RESOLUTION.height; v++)
	{
		for (u = 0; u < lightfield.ANGULAR_RESOLUTION.width; u++)
		{
			subapertureImage = lightfield.getSubapertureImage(u, v);
			differenceImage = subapertureImage - refocusedImage;
			multiply(differenceImage, differenceImage, squaredDifference);
			variance += squaredDifference;
		}
	}
	variance /= lightfield.ANGULAR_RESOLUTION.area();

	Mat standardDeviation, confidence;
	sqrt(variance, standardDeviation);
	filter2D(standardDeviation, confidence, DDEPTH, CORRESPONDENCE_WINDOW,
		WINDOW_CENTER, 0, BORDER_TYPE);
	
	return confidence;
}


Mat CDCDepthEstimator::argMaxAlpha(vector<Mat> responses)
{
	// sort responses until the two greatest values have been found
	Mat mat1, mat2;
	int height = responses[0].size().height;
	int width = responses[0].size().width;
	int responseIndex;
	Vec2i tmp;
	float response1, response2;
	MatIterator_<Vec2f> it1, it2, end1;
	for (int loopIndex = 0; loopIndex < 2; loopIndex++)
	{
		for (responseIndex = 1; responseIndex < responses.size() - loopIndex; responseIndex++)
		{
			mat1 = responses[responseIndex - 1];
			mat2 = responses[responseIndex];

            for( it1 = mat1.begin<Vec2f>(), it2 = mat2.begin<Vec2f>(),
				end1 = mat1.end<Vec2f>(); it1 != end1; ++it1, ++it2 )
            {
				response1	= (*it1)[0];
				response2	= (*it2)[0];

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
	Mat firstExtrema, secondExtrema, alphas;
	Mat in[]		= { responses[firstExtremumIndex],
		responses[secondExtremumIndex] };
	Mat out[]		= { alphas };
	int from_to[]	= { 1,0, 3,1 };
	mixChannels(in, 2, out, 1, from_to, 2);

	return alphas;
}


Mat CDCDepthEstimator::argMinAlpha(vector<Mat> responses)
{
	// sort responses until the two smallest values have been found
	Mat mat1, mat2;
	int height = responses[0].size().height;
	int width = responses[0].size().width;
	int responseIndex;
	Vec2i tmp;
	float response1, response2;
	MatIterator_<Vec2f> it1, it2, end1;
	for (int loopIndex = 0; loopIndex < 2; loopIndex++)
	{
		for (responseIndex = 1; responseIndex < responses.size() - loopIndex; responseIndex++)
		{
			mat1 = responses[responseIndex - 1];
			mat2 = responses[responseIndex];

            for( it1 = mat1.begin<Vec2f>(), it2 = mat2.begin<Vec2f>(),
				end1 = mat1.end<Vec2f>(); it1 != end1; ++it1, ++it2 )
            {
				response1	= (*it1)[0];
				response2	= (*it2)[0];

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
	Mat firstExtrema, secondExtrema, alphas;
	Mat in[]		= { responses[firstExtremumIndex],
		responses[secondExtremumIndex] };
	Mat out[]		= { alphas };
	int from_to[]	= { 1,0, 3,1 };
	mixChannels(in, 2, out, 1, from_to, 2);

	return alphas;
}


Mat CDCDepthEstimator::calculateConfidence(Mat extrema)
{
	Mat firstExtrema, secondExtrema, confidence;

	Mat in[]		= { extrema };
	Mat out[]		= { firstExtrema, secondExtrema };
	int from_to[]	= { 0,0, 1,1 };
	mixChannels(in, 1, out, 2, from_to, 2);

	divide(firstExtrema, secondExtrema, confidence);

	return confidence;
}


Mat CDCDepthEstimator::mrf(Mat depth1, Mat depth2, Mat confidence1, Mat confidence2)
{
	// TODO ...
	Mat depth;
	return depth;
}
