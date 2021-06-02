#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Classification : public TFCore
	{
	public:
		Classification();
		~Classification();

		std::vector<std::vector<std::vector<float>>> GetOutput();
		std::vector<std::vector<float>> GetOutputByOpIndex(int nOutputOpIndex = 0);
		std::vector<std::vector<int>> GetPredCls(float);
		std::vector<int> GetPredClsByOpIndex(float, int nOutputOpIndex = 0);
		void GetPredClsAndSftmx(std::vector<std::vector<int>>&, std::vector<std::vector<float>>&, float);
		void GetPredClsAndSftmxByOpIndex(std::vector<int>&, std::vector<float>&, float, int nOutputOpIndex = 0);
	};
}