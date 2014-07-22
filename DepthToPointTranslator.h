#pragma once

#include <opencv2/ocl/ocl.hpp>
#include "LightFieldPicture.h"

/**
 * The abstract base class for any algorithm that translates reconstructed depth
 * into 3D points.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-22
 */
class DepthToPointTranslator
{
public:
	DepthToPointTranslator(void);
	~DepthToPointTranslator(void);

	virtual oclMat translateDepthToPoints(const oclMat& depth,
		const LightFieldPicture& lightfield) const =0;
};

