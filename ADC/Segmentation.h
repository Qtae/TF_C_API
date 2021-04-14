#pragma once
#include "TFCore.h"


class Segmentation : public TFCore
{
public:
	Segmentation();
	~Segmentation();
	std::vector<std::vector<float *>> GetResult();
};