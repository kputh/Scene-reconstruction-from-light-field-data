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


float ImageRenderer::getAlpha()
{
	return this->alpha;
}


void ImageRenderer::setAlpha(float alpha)
{
	this->alpha = alpha;
}


Vec2i ImageRenderer::getPinholePosition()
{
	return this->pinholePosition;
}


void ImageRenderer::setPinholePosition(Vec2i pinholePosition)
{
	this->pinholePosition = pinholePosition;
}
