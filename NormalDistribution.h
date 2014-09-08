#pragma once

/**
 * A two-dimensional normal distribution. It is used as an approximation of the
 * aperture function by ImageRenderer3.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-09
 */
class NormalDistribution
{
	float mean1, mean2, standardDeviation1, standardDeviation2, cFactor;
public:
	NormalDistribution(float mean1, float mean2,
		float standardDeviation1, float standardDeviation2);
	~NormalDistribution(void);

	float f(float x1, float x2) const;
};

