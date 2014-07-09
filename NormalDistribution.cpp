#define _USE_MATH_DEFINES	// for math constants in C++
#include <cmath>
#include "NormalDistribution.h"


NormalDistribution::NormalDistribution(float mean1, float mean2,
	float standardDeviation1, float standardDeviation2)
{
	this->mean1 = mean1;
	this->mean2 = mean2;
	this->standardDeviation1 = standardDeviation1;
	this->standardDeviation2 = standardDeviation2;
	this->cFactor = 1.0 / (2.0 * M_PI * standardDeviation1 * standardDeviation2);
}


NormalDistribution::~NormalDistribution(void)
{
}


float NormalDistribution::f(float x1, float x2) const
{
	float value1 = (x1 - this->mean1) / this->standardDeviation1;
	float value2 = (x2 - this->mean2) / this->standardDeviation2;
	float result = this->cFactor * exp(-0.5 * (value1 * value1 + value2 * value2));

	return result;
}