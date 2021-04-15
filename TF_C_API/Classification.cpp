#include "Classification.h"

namespace TFTool
{
	Classification::Classification()
	{
	}

	Classification::~Classification()
	{
	}

	std::vector<std::vector<std::vector<float>>> Classification::GetResult()
	{
		std::vector<std::vector<std::vector<float>>> vtResult;
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator 갯수 iteration
		{
			std::vector<std::vector<float>> vtResultOp;
			for (int i = 0; i < vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
			{
				int nBatch = (int)TF_Dim(vtOutputTensors[opsIdx][i], 0);
				int nClass = (int)TF_Dim(vtOutputTensors[opsIdx][i], 1);
				float *output = new float[nBatch * nClass];
				std::memcpy(output, TF_TensorData(vtOutputTensors[opsIdx][i]), nBatch * nClass * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					std::vector<float> vtSoftMax;
					for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
					{
						vtSoftMax.push_back(output[batchIdx * nClass + clsIdx]);
					}
					vtResultOp.push_back(vtSoftMax);
				}
				TF_DeleteTensor(vtOutputTensors[opsIdx][i]);
			}
			vtResult.push_back(vtResultOp);
		}
		return vtResult;
	}
}