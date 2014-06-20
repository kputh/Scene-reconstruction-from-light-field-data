#include "ImageRenderer.h"


ImageRenderer::ImageRenderer(void)
{
}


ImageRenderer::~ImageRenderer(void)
{
}


LightFieldFromLfpFile ImageRenderer::getLightfield()
{
	return this->lightfield;
}


void ImageRenderer::setLightfield(LightFieldFromLfpFile lightfield)
{
	this->lightfield = lightfield;
}


double ImageRenderer::getFocalLength()
{
	return this->focalLength;
}


void ImageRenderer::setFocalLength(double focalLength)
{
	this->focalLength = focalLength;
}


Vec2i ImageRenderer::getPinholePosition()
{
	return this->pinholePosition;
}


void ImageRenderer::setPinholePosition(Vec2i pinholePosition)
{
	this->pinholePosition = pinholePosition;
}
