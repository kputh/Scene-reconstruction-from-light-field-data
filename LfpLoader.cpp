#include "LfpLoader.h"
#include "lfpsplitter.c"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>


const char LfpLoader::IMAGE_KEY[]	= "image";
const char LfpLoader::WIDTH_KEY[]	= "width";
const char LfpLoader::HEIGHT_KEY[]	= "height";


LfpLoader::LfpLoader(void)
{
}

	
LfpLoader::LfpLoader(const string& path)
{
	// 1) split raw file
	const char* cFileName = path.c_str();
	char *period = NULL;
	lfp_file_p lfp = NULL;

	if (!(lfp = lfp_create(cFileName))) {
		lfp_close(lfp);
		throw new std::runtime_error("Failed to open file.");
	}
	
	if (!lfp_file_check(lfp)) {
		lfp_close(lfp);
		throw new std::runtime_error("File is no LFP raw file.");
	}
	
	/*
	// save the first part of the filename to name the jpgs later
	if (!(lfp->filename = strdup(cFileName))) {
		lfp_close(lfp);
		throw new std::runtime_error("Error extracting filename.");
	}
	period = strrchr(lfp->filename,'.');
	if (period) *period = '\0';
	*/
	lfp_parse_sections(lfp);

	// 2) extract image metadata
	int width, height, imageLength;
	char* image;
	rapidjson::Document doc;
	for (lfp_section_p section = lfp->sections; section != NULL; section = section->next)
	{
		switch (section->type) {
			case LFP_RAW_IMAGE:
				image = section->data;
				imageLength = section->len;
				break;
			
			case LFP_JSON:
				doc.Parse<0>(section->data);

				if (doc.HasParseError())
				{
					lfp_close(lfp);
					throw new std::runtime_error("A JSON parsing error occured.");
				}

				// if this is JSON document 1
				if (doc.HasMember(IMAGE_KEY))
				{
					const rapidjson::Value& image = doc[IMAGE_KEY];
					height = image[HEIGHT_KEY].GetInt();
					width = image[WIDTH_KEY].GetInt();

					readMetadata(doc);
				}
				// if this is JSON document 2
				else if (doc["camera"].HasMember("serialNumber"))
				{
					this->cameraSerialNumber = doc["camera"]["serialNumber"].GetString();
				}

				break;
		}
	}
	
	if (width == NULL || height == NULL || image == NULL || imageLength == NULL)
	{
		lfp_close(lfp);
		throw new std::runtime_error("Image metadata not found.");
	}

	// 3) extract image to Mat
	int buflen = 0;
	char *buf;
	buf = converted_image((unsigned char *)image, &buflen, imageLength);
	Mat bayerImage(height, width, CV_16UC1, (unsigned short*) buf);
	lfp_close(lfp);

	this->bayerImage = bayerImage;
}


LfpLoader::~LfpLoader(void)
{
}


void LfpLoader::readMetadata(const rapidjson::Document& doc)
{
	const rapidjson::Value& devices = doc["devices"];
	const rapidjson::Value& mla = devices["mla"];
	const rapidjson::Value& scaleFactor = mla["scaleFactor"];
	const rapidjson::Value& sensorOffset = mla["sensorOffset"];

	this->pixelPitch	= devices["sensor"]["pixelPitch"].GetDouble();
	this->focalLength	= devices["lens"]["focalLength"].GetDouble();
	this->lensPitch		= mla["lensPitch"].GetDouble();
	this->rotationAngle	= mla["rotation"].GetDouble();
	this->scaleFactor	= Vec2d(scaleFactor["x"].GetDouble(),
		scaleFactor["y"].GetDouble());
	this->sensorOffset	= Vec3d(sensorOffset["x"].GetDouble(),
		sensorOffset["y"].GetDouble(),
		sensorOffset["z"].GetDouble());
}
