#include "Segmentation.h"

namespace TFTool
{
	Segmentation::Segmentation()
	{
	}

	Segmentation::~Segmentation()
	{
	}

	std::vector<std::vector<float *>> Segmentation::GetResult()
	{
		std::vector<std::vector<float *>> vtResult;
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator °¹¼ö iteration
		{
			std::vector<float *> vtResultOp;
			for (int i = 0; i < vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
			{
				int nBatch = (int)TF_Dim(vtOutputTensors[opsIdx][i], 0);
				int nOutputSize = (int)TF_Dim(vtOutputTensors[opsIdx][i], 1) * (int)TF_Dim(vtOutputTensors[opsIdx][i], 1);
				float *output = new float[nBatch * nOutputSize];
				std::memcpy(output, TF_TensorData(vtOutputTensors[opsIdx][i]), nBatch * nOutputSize * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					vtResultOp.push_back(output);
				}
			}
			vtResult.push_back(vtResultOp);
		}
		return vtResult;
	}
}