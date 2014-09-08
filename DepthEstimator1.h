
#include "DepthEstimator.h"

/**
 * Unfinished implementation.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-07-08
 */
class DepthEstimator1 :
	public DepthEstimator
{
	static const float ALPHA;
public:
	DepthEstimator1(void);
	~DepthEstimator1(void);

	Mat estimateDepth(const LightFieldPicture lightfield);
};
