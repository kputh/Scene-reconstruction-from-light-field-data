#include "ImageRenderer.h"


ImageRenderer::ImageRenderer(void)
{
}


ImageRenderer::~ImageRenderer(void)
{
}


LightFieldPicture ImageRenderer::getLightfield()
{
	return this->lightfield;
}


void ImageRenderer::setLightfield(LightFieldPicture lightfield)
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
