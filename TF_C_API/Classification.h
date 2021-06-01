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
		std::vector<std::vector<float>> GetOutput(int);
		std::vector<std::vector<int>> GetPredictIndex(float);
		std::vector<int> GetPredictIndex(float, int);
		std::vector<std::vector<int, float>> GetPredictIndexWithSoftmax(float);
		std::vector<int, float> GetPredictIndexWithSoftmax(float, int);
	};
}