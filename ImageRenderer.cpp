#include "ImageRenderer.h"


ImageRenderer::ImageRenderer(void)
{
}


ImageRenderer::~ImageRenderer(void)
{
}


LightFieldPicture ImageRenderer::getLightfield() const
{
	return this->lightfield;
}


void ImageRenderer::setLightfield(const LightFieldPicture& lightfield)
{
	this->lightfield = lightfield;
}


float ImageRenderer::getAlpha() const
{
	return this->alpha;
}


void ImageRenderer::setAlpha(float alpha)
{
	this->alpha = alpha;
}


Vec2i ImageRenderer::getPinholePosition() const
{
	return this->pinholePosition;
}


void ImageRenderer::setPinholePosition(Vec2i pinholePosition)
{
	this->pinholePosition = pinholePosition;
}
