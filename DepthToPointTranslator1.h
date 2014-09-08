#pragma once

#include "DepthToPointTranslator.h"

/**
 * Implementation of an algorithm that translates reconstructed depth into 3D
 * points.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-22
 */
class DepthToPointTranslator1 :
	public DepthToPointTranslator
{
public:
	DepthToPointTranslator1(void);
	~DepthToPointTranslator1(void);

	Mat translateDepthToPoints(const Mat& depth, const Mat& calibrationMatrix,
		const Mat& rotation, const Mat& translation) const;
};

