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
	mModelPath = "";//vtModelPath
	mRunOptions = TF_NewBufferFromString("", 0);
	mSessionOptions = TF_NewSessionOptions();
	mGraph = TF_NewGraph();
	mStatus = TF_NewStatus();
	mMetaGraph = TF_NewBuffer();
	mSession = TF_NewSession(mGraph, mSessionOptions, mStatus);//Session --> 2GB
	const char* TAG = "serve";
	mSession = TF_LoadSessionFromSavedModel(mSessionOptions, mRunOptions, mModelPath, &TAG, 1, mGraph, mMetaGraph, mStatus);//Session --> ~10GB

	mInputOpNum = (int)vtInputOpNames.size();
	mOutputOpNum = (int)vtOutputOpNames.size();

	if (TF_GetCode(mStatus) != TF_OK)
	{
		std::cout << "mStatus : " << TF_Message(mStatus) << std::endl;
		return false;
	}

	mInputOpsArr = new TF_Output[mInputOpNum];
	mOutputOpsArr = new TF_Output[mOutputOpNum];
	mInputDims = new int[mInputOpNum];
	mOutputDims = new int[mOutputOpNum];
	mInputDimsArr = new long long*[mInputOpNum];
	mOutputDimsArr = new long long*[mOutputOpNum];
	mInputDataSizePerBatch = new std::size_t[mInputOpNum];
	mOutputDataSizePerBatch = new std::size_t[mOutputOpNum];

	for (int i = 0; i < mInputOpNum; ++i)
	{
		TF_Operation* inputOp = TF_GraphOperationByName(mGraph, vtInputOpNames[i]);
		if (inputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		mInputOpsArr[i] = TF_Output{ inputOp, 0 };
		mInputDims[i] = TF_GraphGetTensorNumDims(mGraph, mInputOpsArr[0], mStatus);
		int64_t* inputShape = new int64_t[mInputDims[i]];
		TF_GraphGetTensorShape(mGraph, mInputOpsArr[0], inputShape, mInputDims[i], mStatus);
		mInputDimsArr[i] = new long long[mInputDims[i]];
		mInputDimsArr[i][0] = static_cast<long long>(1);
		for (int j = 1; j < mInputDims[i]; ++j) mInputDimsArr[i][j] = static_cast<long long>(inputShape[j]);
		mInputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(mInputOpsArr[i]));
		for (int j = 1; j < mInputDims[i]; ++j) mInputDataSizePerBatch[i] = mInputDataSizePerBatch[i] * static_cast<int>(inputShape[j]);
		delete[] inputShape;
	}

	for (int i = 0; i < mOutputOpNum; ++i)
	{
		TF_Operation* outputOp = TF_GraphOperationByName(mGraph, vtOutputOpNames[i]);
		if (outputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		mOutputOpsArr[i] = TF_Output{ outputOp, 0 };
		mOutputDims[i] = TF_GraphGetTensorNumDims(mGraph, mOutputOpsArr[0], mStatus);
		int64_t* outputShape = new int64_t[mOutputDims[i]];
		TF_GraphGetTensorShape(mGraph, mOutputOpsArr[0], outputShape, mOutputDims[i], mStatus);
		mOutputDimsArr[i] = new long long[mOutputDims[i]];
		mOutputDimsArr[i][0] = static_cast<long long>(1);
		for (int j = 1; j < mOutputDims[i]; ++j) mOutputDimsArr[i][j] = static_cast<long long>(outputShape[j]);
		mOutputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(mOutputOpsArr[0]));
		for (int j = 1; j < mOutputDims[i]; ++j) mOutputDataSizePerBatch[i] = mOutputDataSizePerBatch[i] * static_cast<int>(outputShape[j]);
		delete[] outputShape;
	}

	mIsModelLoaded = true;
	return true;
}

bool Classification::GetOutput(float*** pClassificationResultArray)
{
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)//output operator 갯수 iteration
	{
		int preBatch = 0;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int cls = (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *output = new float[batch * cls];
			std::memcpy(output, TF_TensorData(mOutputTensors[opsIdx][i]), batch * cls * sizeof(float));
			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
				{
					pClassificationResultArray[opsIdx][preBatch + imgIdx][clsIdx] = output[imgIdx * cls + clsIdx];
				}
			}
			preBatch += batch;
		}
	}
	return true;
}

std::vector<std::vector<float>> Classification::GetOutputByOpIndex(int nOutputOpIndex)
{
	std::vector<std::vector<std::vector<float>>> result;
	std::vector<std::vector<float>> resultOp;
	for (int i = 0; i < mOutputTensors[nOutputOpIndex].size(); ++i)
	{
		int batch = (int)TF_Dim(mOutputTensors[nOutputOpIndex][i], 0);
		int cls = (int)TF_Dim(mOutputTensors[nOutputOpIndex][i], 1);
		float *output = new float[batch * cls];
		std::memcpy(output, TF_TensorData(mOutputTensors[nOutputOpIndex][i]), batch * cls * sizeof(float));
		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
		{
			std::vector<float> softMax;
			for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
			{
				softMax.push_back(output[imgIdx * cls + clsIdx]);
			}
			resultOp.push_back(softMax);
		}
	}
	result.push_back(resultOp);
	return result[0];
}

std::vector<std::vector<int>> Classification::GetPredCls(float fThresh)
{
	std::vector<std::vector<int>> result;
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		std::vector<int> resultOp;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int cls = (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *output = new float[batch * cls];
			std::memcpy(output, TF_TensorData(mOutputTensors[opsIdx][i]), batch * cls * sizeof(float));
			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				int nMaxIndex = -1;

				float fMaxValue = 0;
				for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
				{
					if ((output[imgIdx * cls + clsIdx] > fThresh) && (output[imgIdx * cls + clsIdx] > fMaxValue))
					{
						nMaxIndex = clsIdx;
						fMaxValue = output[imgIdx * cls + clsIdx];
					}
				}
				resultOp.push_back(nMaxIndex);
			}
		}
		result.push_back(resultOp);
	}
	return result;
}

std::vector<std::vector<std::vector<float>>> Classification::GetSoftMXResult()
{
	std::vector<std::vector<std::vector<float>>> result;
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		std::vector<std::vector<float>> resultOp;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int cls = (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *output = new float[batch * cls];
			std::memcpy(output, TF_TensorData(mOutputTensors[opsIdx][i]), batch * cls * sizeof(float));

			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				std::vector<float> SoftMXValue;
				for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
				{
					SoftMXValue.push_back(output[imgIdx * cls + clsIdx]);
				}
				resultOp.push_back(SoftMXValue);
			}
		}
		result.push_back(resultOp);
	}
	return result;
}

std::vector<int> Classification::GetPredClsByOpIndex(float fThresh, int nOutputOpIndex)
{
	std::vector<int> result;
	for (int i = 0; i < mOutputTensors[nOutputOpIndex].size(); ++i)
	{
		int batch = (int)TF_Dim(mOutputTensors[nOutputOpIndex][i], 0);
		int cls = (int)TF_Dim(mOutputTensors[nOutputOpIndex][i], 1);
		float *output = new float[batch * cls];
		std::memcpy(output, TF_TensorData(mOutputTensors[nOutputOpIndex][i]), batch * cls * sizeof(float));
		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
		{
			int nMaxIndex = -1;
			float fMaxValue = 0;
			for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
			{
				if ((output[imgIdx * cls + clsIdx] > fThresh) && (output[imgIdx * cls + clsIdx] > fMaxValue))
				{
					nMaxIndex = clsIdx;
					fMaxValue = output[imgIdx * cls + clsIdx];
				}
			}
			result.push_back(nMaxIndex);
		}
	}
	return result;
}

void Classification::GetPredClsAndSftmx(std::vector<std::vector<int>>& vtPredCls, std::vector<std::vector<float>>& vtSftmx, float fThresh)
{
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		std::vector<int> predClsOp;
		std::vector<float> sftmxOp;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int cls = (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *output = new float[batch * cls];
			std::memcpy(output, TF_TensorData(mOutputTensors[opsIdx][i]), batch * cls * sizeof(float));
			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				int nMaxIndex = -1;
				float fMaxValue = 0;
				for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
				{
					if ((output[imgIdx * cls + clsIdx] > fThresh) && (output[imgIdx * cls + clsIdx] > fMaxValue))
					{
						nMaxIndex = clsIdx;
						fMaxValue = output[imgIdx * cls + clsIdx];
					}
				}
				predClsOp.push_back(nMaxIndex);
				sftmxOp.push_back(fMaxValue);
			}
		}
		vtPredCls.push_back(predClsOp);
		vtSftmx.push_back(sftmxOp);
	}
	return;
}

void Classification::GetPredClsAndSftmxByOpIndex(std::vector<int>& vtPredCls, std::vector<float>& vtSftmx, float fThresh, int nOutputOpIndex)
{
	for (int i = 0; i < mOutputTensors[nOutputOpIndex].size(); ++i)
	{
		int batch = (int)TF_Dim(mOutputTensors[nOutputOpIndex][i], 0);
		int cls = (int)TF_Dim(mOutputTensors[nOutputOpIndex][i], 1);
		float *output = new float[batch * cls];
		std::memcpy(output, TF_TensorData(mOutputTensors[nOutputOpIndex][i]), batch * cls * sizeof(float));
		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
		{
			int nMaxIndex = -1;
			float fMaxValue = 0;
			for (int clsIdx = 0; clsIdx < cls; ++clsIdx)
			{
				if ((output[imgIdx * cls + clsIdx] > fThresh) && (output[imgIdx * cls + clsIdx] > fMaxValue))
				{
					nMaxIndex = clsIdx;
					fMaxValue = output[imgIdx * cls + clsIdx];
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