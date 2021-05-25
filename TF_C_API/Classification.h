#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Classification : public TFCore
	{
	public:
		Classification();
		~Classification();
		std::vector<std::vector<std::vector<float>>> GetResult();
	};
}