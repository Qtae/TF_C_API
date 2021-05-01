#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Segmentation : public TFCore
	{
	public:
		__declspec(dllexport) Segmentation();
		__declspec(dllexport) ~Segmentation();
		__declspec(dllexport) std::vector<std::vector<float *>> GetResult();
	};
}