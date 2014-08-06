#pragma once

#include <opencv2/core/core.hpp>
#include "document.h"

using namespace cv;

/**
 * Loads light field data from Light Field Picture files (Lytro's raw.lfp files).
 *
 * Utilites Nirav Patel's lfdsplitter (https://github.com/nrpatel/lfptools) to
 * split the file into it's components and rapidjson
 * (https://code.google.com/p/rapidjson/) to extract image metadata.
 *
 * @author      Kai Puth <kai.puth@student.htw-berlin.de>
 * @version     0.1
 * @since       2014-05-13
 */
class LfpLoader
{
	static const char IMAGE_KEY[];
	static const char WIDTH_KEY[];
	static const char HEIGHT_KEY[];

	void readMetadata(const rapidjson::Document& doc);

public:
	Mat bayerImage;

	double pixelPitch;
	double focalLength;
	double lensPitch;
	double rotationAngle;
	Vec2d scaleFactor;
	Vec3d sensorOffset;
	string cameraSerialNumber;
	int black, white;
	double gamma;
	Mat whiteBalancingMatrix, colorCorrectionMatrix;

	LfpLoader(void);
	LfpLoader(const string& pathToFile);
	~LfpLoader(void);
};