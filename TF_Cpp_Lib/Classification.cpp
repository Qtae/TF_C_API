#pragma once
#include "Classification.h"


Classification::Classification()
{
}

Classification::~Classification()
{
}

bool Classification::LoadEnsembleModel(std::vector<const char*> vtModelPath, std::vector<const char*>& vtInputOpNames, std::vector<const char*>& vtOutputOpNames)
{
	for (std::vector<const char*>::iterator it = vtModelPath.begin(); it != vtModelPath.end(); ++it)
	{
		LoadModel(*it, vtInputOpNames, vtOutputOpNames);
	}
	m_ModelPath = "";//vtModelPath
	m_RunOptions = TF_NewBufferFromString("", 0);
	m_SessionOptions = TF_NewSessionOptions();
	m_Graph = TF_NewGraph();
	m_Status = TF_NewStatus();
	m_MetaGraph = TF_NewBuffer();
	m_Session = TF_NewSession(m_Graph, m_SessionOptions, m_Status);//Session --> 2GB
	const char* tag = "serve";
	m_Session = TF_LoadSessionFromSavedModel(m_SessionOptions, m_RunOptions, m_ModelPath, &tag, 1, m_Graph, m_MetaGraph, m_Status);//Session --> ~10GB

	m_nInputOps = (int)vtInputOpNames.size();
	m_nOutputOps = (int)vtOutputOpNames.size();

	if (TF_GetCode(m_Status) != TF_OK)
	{
		std::cout << "m_Status : " << TF_Message(m_Status) << std::endl;
		return false;
	}

	m_arrInputOps = new TF_Output[m_nInputOps];
	m_arrOutputOps = new TF_Output[m_nOutputOps];
	m_nInputDims = new int[m_nInputOps];
	m_nOutputDims = new int[m_nOutputOps];
	m_InputDims = new long long*[m_nInputOps];
	m_OutputDims = new long long*[m_nOutputOps];
	m_InputDataSizePerBatch = new std::size_t[m_nInputOps];
	m_OutputDataSizePerBatch = new std::size_t[m_nOutputOps];

	for (int i = 0; i < m_nInputOps; ++i)
	{
		TF_Operation* InputOp = TF_GraphOperationByName(m_Graph, vtInputOpNames[i]);
		if (InputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		m_arrInputOps[i] = TF_Output{ InputOp, 0 };
		m_nInputDims[i] = TF_GraphGetTensorNumDims(m_Graph, m_arrInputOps[0], m_Status);
		int64_t* InputShape = new int64_t[m_nInputDims[i]];
		TF_GraphGetTensorShape(m_Graph, m_arrInputOps[0], InputShape, m_nInputDims[i], m_Status);
		m_InputDims[i] = new long long[m_nInputDims[i]];
		m_InputDims[i][0] = static_cast<long long>(1);
		for (int j = 1; j < m_nInputDims[i]; ++j) m_InputDims[i][j] = static_cast<long long>(InputShape[j]);
		m_InputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(m_arrInputOps[i]));
		for (int j = 1; j < m_nInputDims[i]; ++j) m_InputDataSizePerBatch[i] = m_InputDataSizePerBatch[i] * static_cast<int>(InputShape[j]);
		delete[] InputShape;
	}

	for (int i = 0; i < m_nOutputOps; ++i)
	{
		TF_Operation* OutputOp = TF_GraphOperationByName(m_Graph, vtOutputOpNames[i]);
		if (OutputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		m_arrOutputOps[i] = TF_Output{ OutputOp, 0 };
		m_nOutputDims[i] = TF_GraphGetTensorNumDims(m_Graph, m_arrOutputOps[0], m_Status);
		int64_t* OutputShape = new int64_t[m_nOutputDims[i]];
		TF_GraphGetTensorShape(m_Graph, m_arrOutputOps[0], OutputShape, m_nOutputDims[i], m_Status);
		m_OutputDims[i] = new long long[m_nOutputDims[i]];
		m_OutputDims[i][0] = static_cast<long long>(1);
		for (int j = 1; j < m_nOutputDims[i]; ++j) m_OutputDims[i][j] = static_cast<long long>(OutputShape[j]);
		m_OutputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(m_arrOutputOps[0]));
		for (int j = 1; j < m_nOutputDims[i]; ++j) m_OutputDataSizePerBatch[i] = m_OutputDataSizePerBatch[i] * static_cast<int>(OutputShape[j]);
		delete[] OutputShape;
	}
		
	m_bModelLoaded = true;
	return true;
}

bool Classification::GetOutput(float*** pClassificationResultArray)
{
	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator 갯수 iteration
	{
		int nPreBatch = 0;
		for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
			int nClass = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
			float *output = new float[nBatch * nClass];
			std::memcpy(output, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nClass * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
				{
					pClassificationResultArray[opsIdx][nPreBatch + batchIdx][clsIdx] = output[batchIdx * nClass + clsIdx];
				}
			}
			nPreBatch += nBatch;
		}
	}
	return true;
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

std::vector<std::vector<int>> Classification::GetHardVoteEnsembleOutput()
{
	std::vector<std::vector<int>> res;
	return res;
}

std::vector<std::vector<int>> Classification::GetSoftVoteEnsembleOutput()
{
	std::vector<std::vector<int>> res;
	return res;
}