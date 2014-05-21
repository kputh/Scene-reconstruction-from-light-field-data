#include "LfpLoader.h"
#include "lfpsplitter.c"
#include "rapidjson/document.h"


const char LfpLoader::IMAGE_KEY[] = "image";
const char LfpLoader::WIDTH_KEY[] = "width";
const char LfpLoader::HEIGHT_KEY[] = "height";


Mat LfpLoader::load(const string& path)
{
	// 1) split raw file
	const char* cFileName = path.c_str();
    char *period = NULL;
    lfp_file_p lfp = NULL;

    if (!(lfp = lfp_create(cFileName))) {
		// TODO
		lfp_close(lfp);
		throw new std::runtime_error("Failed to open file.");
    }
    
    if (!lfp_file_check(lfp)) {
		// TODO
		lfp_close(lfp);
		throw new std::runtime_error("File is no LFP raw file.");
    }
    
    // save the first part of the filename to name the jpgs later
    if (!(lfp->filename = strdup(cFileName))) {
        lfp_close(lfp);
		throw new std::runtime_error("Error extracting filename.");
    }
    period = strrchr(lfp->filename,'.');
    if (period) *period = '\0';

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
	char *buf;
    int buflen = 0;
	buf = converted_image((unsigned char *)image, &buflen, imageLength);

	Mat imageMat = Mat(height, width, CV_8UC1);		// TODO: stimmt die Bit-Tiefe?
	int rowIndex, colIndex;
	for (int i = 0; i < buflen; i += 3) {
		colIndex = i / width;
		rowIndex = i % width;
		imageMat.at<int>(rowIndex, colIndex) = buf[i];
	}

	lfp_close(lfp);
    return imageMat;
}