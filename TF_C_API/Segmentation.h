#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Segmentation : public TFCore
	{
	public:
		Segmentation();
		~Segmentation();
		std::vector<std::vector<float *>> GetResult();
	};
}