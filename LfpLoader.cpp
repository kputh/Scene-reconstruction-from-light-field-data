#include "LfpLoader.h"
#include "lfpsplitter.c"
#include "rapidjson/document.h"
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

				if (doc.HasMember(IMAGE_KEY))
				{
					height = doc[IMAGE_KEY][HEIGHT_KEY].GetInt();
					width = doc[IMAGE_KEY][WIDTH_KEY].GetInt();
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

	// TODO read metadata from file
	/*
	this->pixelPitch	= doc["devices"]["sensor"]["pixelPitch"].GetDouble();
	this->lensPitch		= doc["devices"]["mla"]["lensPitch"].GetDouble();
	this->rotationAngle	= doc["devices"]["mla"]["rotation"].GetDouble();
	this->scaleFactorX	= doc["devices"]["mla"]["scaleFactor"]["x"].GetDouble();
	this->scaleFactorY	= doc["devices"]["mla"]["scaleFactor"]["y"].GetDouble();
	this->sensorOffsetX	= doc["devices"]["mla"]["sensorOffset"]["x"].GetDouble();
	this->sensorOffsetY	= doc["devices"]["mla"]["sensorOffset"]["y"].GetDouble();
	*/

	this->pixelPitch	= 0.0000013999999761581417;
	this->focalLength	= 0.0068200001716613766;
	this->lensPitch		= 0.00001389859962463379;
	this->rotationAngle	= 0.002145454753190279;
	this->scaleFactor	= Vec2d(1.0, 1.0014984607696533);
	this->sensorOffset	= Vec3d(0.0000018176757097244258, -0.0000040150876045227051, 0.000025);
}


LfpLoader::~LfpLoader(void)
{
}
