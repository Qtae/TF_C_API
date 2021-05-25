#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Segmentation : public TFCore
	{
	public:
		Segmentation();
		~Segmentation();
		std::vector<std::vector<float*>> GetOutput();

	private:
		bool FreeOutputMap();

	private:
		std::vector<std::vector<float*>> m_vtOutputRes;
	};
}