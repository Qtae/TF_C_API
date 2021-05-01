#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Classification : public TFCore
	{
	public:
		__declspec(dllexport) Classification();
		__declspec(dllexport) ~Classification();
		__declspec(dllexport) std::vector<std::vector<std::vector<float>>> GetResult();
	};
}