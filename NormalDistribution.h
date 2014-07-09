#pragma once
class NormalDistribution
{
	float mean1, mean2, standardDeviation1, standardDeviation2, cFactor;
public:
	NormalDistribution(float mean1, float mean2,
		float standardDeviation1, float standardDeviation2);
	~NormalDistribution(void);

	float f(float x1, float x2) const;
};

