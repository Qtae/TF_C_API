#include "Classification.h"

namespace TFTool
{
	Classification::Classification()
	{
	}

	Classification::~Classification()
	{
	}

	std::vector<std::vector<std::vector<float>>> Classification::GetOutput()
	{
		std::vector<std::vector<std::vector<float>>> vtResult;
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator 갯수 iteration
		{
			std::vector<std::vector<float>> vtResultOp;
			for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
			{
				int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
				int nClass = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
				float *output = new float[nBatch * nClass];
				std::memcpy(output, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nClass * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					std::vector<float> vtSoftMax;
					for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
					{
						vtSoftMax.push_back(output[batchIdx * nClass + clsIdx]);
					}
					vtResultOp.push_back(vtSoftMax);
				}
			}
			vtResult.push_back(vtResultOp);
		}
		return vtResult;
	}

	std::vector<std::vector<float>> Classification::GetOutputByOpIndex(int nOutputOpIndex)
	{
		std::vector<std::vector<std::vector<float>>> vtResult;
		std::vector<std::vector<float>> vtResultOp;
		for (int i = 0; i < m_vtOutputTensors[nOutputOpIndex].size(); ++i)
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[nOutputOpIndex][i], 0);
			int nClass = (int)TF_Dim(m_vtOutputTensors[nOutputOpIndex][i], 1);
			float *output = new float[nBatch * nClass];
			std::memcpy(output, TF_TensorData(m_vtOutputTensors[nOutputOpIndex][i]), nBatch * nClass * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				std::vector<float> vtSoftMax;
				for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
				{
					vtSoftMax.push_back(output[batchIdx * nClass + clsIdx]);
				}
				vtResultOp.push_back(vtSoftMax);
			}
		}
		vtResult.push_back(vtResultOp);
		return vtResult[0];
	}

	std::vector<std::vector<int>> Classification::GetPredCls(float fThresh)
	{
		std::vector<std::vector<int>> vtResult;
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			std::vector<int> vtResultOp;
			for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)
			{
				int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
				int nClass = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
				float *output = new float[nBatch * nClass];
				std::memcpy(output, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nClass * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					int nMaxIndex = -1;
					float fMaxValue = 0;
					for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
					{
						if ((output[batchIdx * nClass + clsIdx] > fThresh) && (output[batchIdx * nClass + clsIdx] > fMaxValue))
						{
							nMaxIndex = clsIdx;
							fMaxValue = output[batchIdx * nClass + clsIdx];
						}
					}
					vtResultOp.push_back(nMaxIndex);
				}
			}
			vtResult.push_back(vtResultOp);
		}
		return vtResult;
	}

	std::vector<int> Classification::GetPredClsByOpIndex(float fThresh, int nOutputOpIndex)
	{
		std::vector<int> vtResult;
		for (int i = 0; i < m_vtOutputTensors[nOutputOpIndex].size(); ++i)
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[nOutputOpIndex][i], 0);
			int nClass = (int)TF_Dim(m_vtOutputTensors[nOutputOpIndex][i], 1);
			float *output = new float[nBatch * nClass];
			std::memcpy(output, TF_TensorData(m_vtOutputTensors[nOutputOpIndex][i]), nBatch * nClass * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				int nMaxIndex = -1;
				float fMaxValue = 0;
				for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
				{
					if ((output[batchIdx * nClass + clsIdx] > fThresh) && (output[batchIdx * nClass + clsIdx] > fMaxValue))
					{
						nMaxIndex = clsIdx;
						fMaxValue = output[batchIdx * nClass + clsIdx];
					}
				}
				vtResult.push_back(nMaxIndex);
			}
		}
		return vtResult;
	}

	void Classification::GetPredClsAndSftmx(std::vector<std::vector<int>>& vtPredCls, std::vector<std::vector<float>>& vtSftmx, float fThresh)
	{
		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			std::vector<int> vtPredClsOp;
			std::vector<float> vtSftmxOp;
			for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)
			{
				int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
				int nClass = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
				float *output = new float[nBatch * nClass];
				std::memcpy(output, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nClass * sizeof(float));
				for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
				{
					int nMaxIndex = -1;
					float fMaxValue = 0;
					for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
					{
						if ((output[batchIdx * nClass + clsIdx] > fThresh) && (output[batchIdx * nClass + clsIdx] > fMaxValue))
						{
							nMaxIndex = clsIdx;
							fMaxValue = output[batchIdx * nClass + clsIdx];
						}
					}
					vtPredClsOp.push_back(nMaxIndex);
					vtSftmxOp.push_back(fMaxValue);
				}
			}
			vtPredCls.push_back(vtPredClsOp);
			vtSftmx.push_back(vtSftmxOp);
		}
		return;
	}

	void Classification::GetPredClsAndSftmxByOpIndex(std::vector<int>& vtPredCls, std::vector<float>& vtSftmx, float fThresh, int nOutputOpIndex)
	{
		for (int i = 0; i < m_vtOutputTensors[nOutputOpIndex].size(); ++i)
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[nOutputOpIndex][i], 0);
			int nClass = (int)TF_Dim(m_vtOutputTensors[nOutputOpIndex][i], 1);
			float *output = new float[nBatch * nClass];
			std::memcpy(output, TF_TensorData(m_vtOutputTensors[nOutputOpIndex][i]), nBatch * nClass * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				int nMaxIndex = -1;
				float fMaxValue = 0;
				for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
				{
					if ((output[batchIdx * nClass + clsIdx] > fThresh) && (output[batchIdx * nClass + clsIdx] > fMaxValue))
					{
						nMaxIndex = clsIdx;
						fMaxValue = output[batchIdx * nClass + clsIdx];
					}
				}
				vtPredCls.push_back(nMaxIndex);
				vtSftmx.push_back(fMaxValue);
			}
		}
		return;
	}
}