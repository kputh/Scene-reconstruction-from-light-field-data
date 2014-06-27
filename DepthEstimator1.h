
#include "DepthEstimator.h"
class DepthEstimator1 :
	public DepthEstimator
{
	static const float FOCAL_LENGTH;
public:
	DepthEstimator1(void);
	~DepthEstimator1(void);

	Mat estimateDepth(const LightFieldPicture lightfield);
};
