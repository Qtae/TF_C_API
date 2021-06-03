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
			std::vector<float*> vtResultOp;
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

	std::vector<std::vector<int*>> Segmentation::GetWholeClsMask()
	{
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator °¹¼ö iteration
		{
			std::vector<int*> vtClassMaskOp;
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
				}
				delete[] pOutput;
			}
			m_vtClassMask.push_back(vtClassMaskOp);
		}
		return m_vtClassMask;
	}

	std::vector<std::vector<int*>> Segmentation::GetBinaryMaskWithClsIndex(int nClsIndex)
	{
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			std::vector<int*> vtClassMaskOp;
			for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
			{
				int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
				int nOutputSize = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1) * (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
				float *pOutput = new float[nBatch * nOutputSize];
				std::memcpy(pOutput, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nOutputSize * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					int* pOutputBatch = new int[nOutputSize];
					for (int pixIdx = 0; pixIdx < nOutputSize; ++pixIdx)
					{
						if ((int)pOutput[batchIdx * nOutputSize + pixIdx] == nClsIndex)
							pOutputBatch[pixIdx] = 1;
						else
							pOutputBatch[pixIdx] = 0;
					}
					vtClassMaskOp.push_back(pOutputBatch);
				}
				delete[] pOutput;
			}
			m_vtClassMask.push_back(vtClassMaskOp);
		}
		return m_vtClassMask;
	}
}