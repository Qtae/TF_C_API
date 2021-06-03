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
		std::vector<std::vector<int*>> GetWholeClsMask();
		std::vector<std::vector<int*>> GetBinaryMaskWithClsIndex(int);

	private:
		bool FreeOutputMap();

	private:
		std::vector<std::vector<float*>> m_vtOutputRes;
		std::vector<std::vector<int*>> m_vtClassMask;
	};
}