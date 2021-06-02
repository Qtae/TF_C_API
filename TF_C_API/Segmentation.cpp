#include "Segmentation.h"

namespace TFTool
{
	Segmentation::Segmentation()
	{
	}

	Segmentation::~Segmentation()
	{
	}

	std::vector<std::vector<float *>> Segmentation::GetOutput()
	{
		if (!m_vtOutputRes.empty())
			FreeOutputMap();

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator °¹¼ö iteration
		{
			std::vector<float *> vtResultOp;
			for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
			{
				int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
				int nOutputSize = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1) * (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
				float *pOutput = new float[nBatch * nOutputSize];
				std::memcpy(pOutput, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nOutputSize * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					float* pOutputBatch = new float[nOutputSize];
					std::memcpy(pOutputBatch, pOutput + batchIdx * nOutputSize, nOutputSize * sizeof(float));
					vtResultOp.push_back(pOutputBatch);
				}
				delete[] pOutput;
			}
			m_vtOutputRes.push_back(vtResultOp);
		}
		return m_vtOutputRes;
	}

	bool Segmentation::FreeOutputMap()
	{
		if (!m_vtOutputRes.empty())
		{
			for (int opsIdx = 0; opsIdx < m_vtOutputRes.size(); ++opsIdx)
			{
				for (int nIdx = 0; nIdx < m_vtOutputRes[opsIdx].size(); ++nIdx)
				{
					delete[] m_vtOutputRes[opsIdx][nIdx];
				}
			}
		}
		return true;
	}
}