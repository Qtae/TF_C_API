#pragma once
#include "TFCore.h"


TFCore::TFCore()
{
	mSession = nullptr;
	mRunOptions = nullptr;
	mSessionOptions = nullptr;
	mGraph = nullptr;
	mStatus = nullptr;
	mMetaGraph = nullptr;
	mInputOpsArr = nullptr;
	mOutputOpsArr = nullptr;
	mInputDims = nullptr;
	mOutputDims = nullptr;
	mInputDimsArr = nullptr;
	mOutputDimsArr = nullptr;
	mInputDataSizePerBatch = nullptr;
	mOutputDataSizePerBatch = nullptr;

	mIsModelLoaded = false;
	mIsDataLoaded = false;
	mbRun = false;
}

TFCore::~TFCore()
{
}

TF_SessionOptions* CreateSessionOptions(double percentage)
{
	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();

	uint8_t config[13] = { 0x32, 0xb, 0x9, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xee, 0x3f, 0x20, 0x1 };

	//uint8_t* bytes = reinterpret_cast<uint8_t*>(&percentage);

	//for (int i = 0; i < sizeof(percentage); i++)
	//{
	//	config[i + 3] = bytes[i];
	//}

	TF_SetConfig(options, (void *)config, 13, status);

	if (TF_GetCode(status) != TF_OK) {
		std::cerr << "Can't set options: " << TF_Message(status) << std::endl;

		TF_DeleteStatus(status);
		return nullptr;
	}

	TF_DeleteStatus(status);
	return options;
}

bool TFCore::LoadModel(const char* modelPath, std::vector<const char*> &inputOpNames, std::vector<const char*>& outputOpNames)
{
	mModelPath = modelPath;
	mRunOptions = TF_NewBufferFromString("", 0);
	//mSessionOptions = CreateSessionOptions(0.99);
	mSessionOptions = TF_NewSessionOptions();
	mGraph = TF_NewGraph();
	mStatus = TF_NewStatus();
	mMetaGraph = TF_NewBuffer();
	mSession = TF_NewSession(mGraph, mSessionOptions, mStatus);
	const char* TAG = "serve";
	mSession = TF_LoadSessionFromSavedModel(mSessionOptions, mRunOptions, mModelPath, &TAG, 1, mGraph, mMetaGraph, mStatus);

	mInputOpNum = (int)inputOpNames.size();
	mOutputOpNum = (int)outputOpNames.size();

	mInputOpNames = inputOpNames;
	mOutputOpNames = outputOpNames;

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
		char inputOpFullName[200];
		strcpy_s(inputOpFullName, sizeof(inputOpFullName), (char*)mInputOpNames[i]);
		char* chIndex = NULL;
		const char* inputOpName = strtok_s(inputOpFullName, ":", &chIndex);
		int inputOpIdx = atoi(chIndex);
		TF_Operation* inputOp = TF_GraphOperationByName(mGraph, inputOpName);
		if (inputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		mInputOpsArr[i] = TF_Output{ inputOp, inputOpIdx };
		mInputDims[i] = TF_GraphGetTensorNumDims(mGraph, mInputOpsArr[i], mStatus);
		int64_t* inputShape = new int64_t[mInputDims[i]];
		TF_GraphGetTensorShape(mGraph, mInputOpsArr[i], inputShape, mInputDims[i], mStatus);
		mInputDimsArr[i] = new long long[mInputDims[i]];
		mInputDimsArr[i][0] = static_cast<long long>(1);
		for (int j = 1; j < mInputDims[i]; ++j) mInputDimsArr[i][j] = static_cast<long long>(inputShape[j]);
		mInputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(mInputOpsArr[i]));
		for (int j = 1; j < mInputDims[i]; ++j) mInputDataSizePerBatch[i] = mInputDataSizePerBatch[i] * static_cast<int>(inputShape[j]);
		delete[] inputShape;
	}

	for (int i = 0; i < mOutputOpNum; ++i)
	{
		char outputOpFullName[200];
		strcpy_s(outputOpFullName, sizeof(outputOpFullName), (char*)mOutputOpNames[i]);
		char* chIndex = NULL;
		const char* outputOpName = strtok_s(outputOpFullName, ":", &chIndex);
		int outputOpIdx = atoi(chIndex);
		TF_Operation* outputOp = TF_GraphOperationByName(mGraph, outputOpName);
		if (outputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		mOutputOpsArr[i] = TF_Output{ outputOp, outputOpIdx };
		mOutputDims[i] = TF_GraphGetTensorNumDims(mGraph, mOutputOpsArr[i], mStatus);
		int64_t* outputShape = new int64_t[mOutputDims[i]];
		TF_GraphGetTensorShape(mGraph, mOutputOpsArr[i], outputShape, mOutputDims[i], mStatus);
		mOutputDimsArr[i] = new long long[mOutputDims[i]];
		mOutputDimsArr[i][0] = static_cast<long long>(1);
		for (int j = 1; j < mOutputDims[i]; ++j) mOutputDimsArr[i][j] = static_cast<long long>(outputShape[j]);
		mOutputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(mOutputOpsArr[i]));
		for (int j = 1; j < mOutputDims[i]; ++j) mOutputDataSizePerBatch[i] = mOutputDataSizePerBatch[i] * static_cast<int>(outputShape[j]);
		delete[] outputShape;
	}
	mIsModelLoaded = true;
	return true;
}

bool TFCore::ReloadModel()
{
	TF_DeleteGraph(mGraph);
	if (mSessionOptions != nullptr) TF_DeleteSessionOptions(mSessionOptions);
	if (mMetaGraph != nullptr) TF_DeleteBuffer(mMetaGraph);
	if (mGraph != nullptr) TF_DeleteGraph(mGraph);
	if (mSession != nullptr)
	{
		TF_CloseSession(mSession, mStatus);
		TF_DeleteSession(mSession, mStatus);
	}
	if (mStatus != nullptr) TF_DeleteStatus(mStatus);

	mSessionOptions = TF_NewSessionOptions();
	mGraph = TF_NewGraph();
	mMetaGraph = TF_NewBuffer();
	mStatus = TF_NewStatus();
	mSession = TF_NewSession(mGraph, mSessionOptions, mStatus);

	const char* TAG = "serve";
	mSession = TF_LoadSessionFromSavedModel(mSessionOptions, mRunOptions, mModelPath, &TAG, 1, mGraph, mMetaGraph, mStatus);

	if (TF_GetCode(mStatus) != TF_OK)
	{
		std::cout << "mStatus : " << TF_Message(mStatus) << std::endl;
		return false;
	}

	return true;
}

bool TFCore::Run(float*** inputImgArr, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor*[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor*[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}

	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		float* imgData = new float[imgNum * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3]];
		if (bNormalize)
		{
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = inputImgArr[imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] / float(255.);
					}
				}
			}
		}
		else
		{
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = inputImgArr[imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx];
					}
				}
			}
		}
		mInputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t inputDataSize = mInputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
			mInputDimsArr[opsIdx],
			mInputDims[opsIdx],
			imgData,
			inputDataSize,
			dealloc,
			nullptr);
		inputTensorArr[opsIdx] = inputImgTensor;

		delete[] imgData;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);
		outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
	}

	TF_SessionRun(mSession, mRunOptions,
		mInputOpsArr, inputTensorArr, mInputOpNum,
		mOutputOpsArr, outputTensorArr, mOutputOpNum,
		nullptr, 0, nullptr, mStatus);

	//Input Tensor 메모리 해제
	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		TF_DeleteTensor(inputTensorArr[opsIdx]);
	}

	if (TF_GetCode(mStatus) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

	return true;
}

bool TFCore::Run(float** inputImgArr, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor*[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor*[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	float* imgData = new float[imgNum * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3]];
	if (bNormalize)
	{
		for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
		{
			for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
			{
				for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
				{
					imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = inputImgArr[imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx] / float(255.);
				}
			}
		}
	}
	else
	{
		for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
		{
			for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
			{
				for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
				{
					imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = inputImgArr[imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx];
				}
			}
		}
	}
	mInputDimsArr[0][0] = static_cast<long long>(imgNum);
	size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(imgNum);

	TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
		mInputDimsArr[0],
		mInputDims[0],
		imgData,
		inputDataSize,
		dealloc,
		nullptr);

	inputTensorArr[0] = inputImgTensor;

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
	}

	TF_SessionRun(mSession, mRunOptions,
		mInputOpsArr, inputTensorArr, mInputOpNum,
		mOutputOpsArr, outputTensorArr, mOutputOpNum,
		nullptr, 0, nullptr, mStatus);

	//Input Tensor 메모리 해제
	TF_DeleteTensor(inputTensorArr[0]);

	if (TF_GetCode(mStatus) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

	return true;
}

bool TFCore::Run(unsigned char*** inputImgArr, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor *[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor *[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		float* imgData = new float[imgNum * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[0][3]];
		if (bNormalize)
		{
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = (float)(inputImgArr[imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx]) / (float)(255.);
					}
				}
			}
		}
		else
		{
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = (float)(inputImgArr[imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx]);
					}
				}
			}
		}
		mInputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t inputDataSize = mInputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
			mInputDimsArr[opsIdx],
			mInputDims[opsIdx],
			imgData,
			inputDataSize,
			dealloc,
			nullptr);
		inputTensorArr[opsIdx] = inputImgTensor;

		delete[] imgData;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
	}

	TF_SessionRun(mSession, mRunOptions,
		mInputOpsArr, inputTensorArr, mInputOpNum,
		mOutputOpsArr, outputTensorArr, mOutputOpNum,
		nullptr, 0, nullptr, mStatus);

	//Input Tensor 메모리 해제
	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		TF_DeleteTensor(inputTensorArr[opsIdx]);
	}

	if (TF_GetCode(mStatus) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

	return true;
}

bool TFCore::Run(unsigned char** inputImgArr, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor *[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor *[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	float* imgData = new float[imgNum * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3]];
	if (bNormalize)
	{
		for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
		{
			for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
			{
				for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
				{
					imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = (float)(inputImgArr[imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx]) / (float)(255.);
				}
			}
		}
	}
	else
	{
		for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
		{
			for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
			{
				for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
				{
					imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = (float)(inputImgArr[imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx]);
				}
			}
		}
	}
	mInputDimsArr[0][0] = static_cast<long long>(imgNum);
	size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(imgNum);

	TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
		mInputDimsArr[0],
		mInputDims[0],
		imgData,
		inputDataSize,
		dealloc,
		nullptr);

	inputTensorArr[0] = inputImgTensor;

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
	}

	TF_SessionRun(mSession, mRunOptions,
		mInputOpsArr, inputTensorArr, mInputOpNum,
		mOutputOpsArr, outputTensorArr, mOutputOpNum,
		nullptr, 0, nullptr, mStatus);

	//Input Tensor 메모리 해제
	TF_DeleteTensor(inputTensorArr[0]);

	if (TF_GetCode(mStatus) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

	return true;
}

bool TFCore::Run(std::vector<cv::Mat> inputImgArr, int, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(sizeof(inputImgArr) / sizeof(float*));  //

	TF_Tensor** inputTensorArr = new TF_Tensor *[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor *[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	for (int opsIdx = 0; opsIdx < inputImgArr.size(); opsIdx++)
	{
		cv::Mat tempMatArray = inputImgArr[opsIdx];

		auto const dealloc = [](void*, std::size_t, void*) {};

		float* imgData = new float[imgNum * tempMatArray.rows * tempMatArray.cols * tempMatArray.channels()];

		if (bNormalize)
		{
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < tempMatArray.rows * tempMatArray.cols * tempMatArray.channels(); ++pixIdx)
				{
					imgData[imgIdx * tempMatArray.rows * tempMatArray.cols * tempMatArray.channels() + pixIdx] = (float)(tempMatArray.at<int>(imgIdx, pixIdx) / (float)(255.));
				}
			}
		}
		else
		{
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < tempMatArray.rows * tempMatArray.cols * tempMatArray.channels(); ++pixIdx)
				{
					imgData[imgIdx * tempMatArray.rows * tempMatArray.cols * tempMatArray.channels() + pixIdx] = (float)(tempMatArray.at<int>(imgIdx, pixIdx));
				}
			}
		}
		mInputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		mInputDimsArr[opsIdx][1] = static_cast<long long>(tempMatArray.rows);
		mInputDimsArr[opsIdx][2] = static_cast<long long>(tempMatArray.cols);
		mInputDimsArr[opsIdx][3] = static_cast<long long>(tempMatArray.channels());

		size_t inputDataSize = mInputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
			mInputDimsArr[opsIdx],
			mInputDims[opsIdx],
			imgData,
			inputDataSize,
			dealloc,
			nullptr);
		inputTensorArr[opsIdx] = inputImgTensor;

		delete[] imgData;

	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
		size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);

		outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
	}

	TF_SessionRun(mSession, mRunOptions,
		mInputOpsArr, inputTensorArr, mInputOpNum,
		mOutputOpsArr, outputTensorArr, mOutputOpNum,
		nullptr, 0, nullptr, mStatus);

	//Input Tensor 메모리 해제
	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		TF_DeleteTensor(inputTensorArr[opsIdx]);
	}

	if (TF_GetCode(mStatus) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

	return true;
}

bool TFCore::Run(unsigned char** inputImg, CPoint imgSize, CPoint cropSize, CPoint overlapSize, CPoint buffPos, bool bNormalize, bool bConvertGrayToColor, bool bReloadEveryRun)
//VisionWorks image input format, has only one input operator
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	if ((cropSize.x <= overlapSize.x) || (cropSize.y <= overlapSize.y))
	{
		std::cout << "Crop size must be larger than overlap size." << std::endl;
		return false;
	}

	TF_Tensor** inputTensorArr = new TF_Tensor*[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor*[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	mImageSize = imgSize;
	mCropSize = cropSize;
	mOverlapSize = overlapSize;

	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - (mCropSize.x - mOverlapSize.x) * (itX - 1) > mCropSize.x) ++itX;
	if (mImageSize.y - (mCropSize.y - mOverlapSize.y) * (itY - 1) > mCropSize.y) ++itY;

	int imgNum = itX * itY;
	int nImageChannel = (int)(mInputDimsArr[0][3]);

	if (bConvertGrayToColor)
	{
		if (bNormalize)
		{
			float* imgData = new float[imgNum * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				int currImgIdx = imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]) / float(255.);
						}
					}
				}
			}

			mInputDimsArr[0][0] = static_cast<long long>(imgNum);
			size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(imgNum);

			TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
				mInputDimsArr[0],
				mInputDims[0],
				imgData,
				inputDataSize,
				dealloc,
				nullptr);

			inputTensorArr[0] = inputImgTensor;

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			{
				mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
				size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);
				outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
			}

			if (bReloadEveryRun && mbRun) ReloadModel();
			TF_SessionRun(mSession, mRunOptions,
				mInputOpsArr, inputTensorArr, mInputOpNum,
				mOutputOpsArr, outputTensorArr, mOutputOpNum,
				nullptr, 0, nullptr, mStatus);
			mbRun = true;

			if (TF_GetCode(mStatus) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

			//Free Memory
			delete[] imgData;
			TF_DeleteTensor(inputTensorArr[0]);
		}
		else
		{
			float* imgData = new float[imgNum * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				int currImgIdx = imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]);
						}
					}
				}
			}

			mInputDimsArr[0][0] = static_cast<long long>(imgNum);
			size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(imgNum);

			TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
				mInputDimsArr[0],
				mInputDims[0],
				imgData,
				inputDataSize,
				dealloc,
				nullptr);

			inputTensorArr[0] = inputImgTensor;

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			{
				mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
				size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);
				outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
			}

			if (bReloadEveryRun && mbRun) ReloadModel();
			TF_SessionRun(mSession, mRunOptions,
				mInputOpsArr, inputTensorArr, mInputOpNum,
				mOutputOpsArr, outputTensorArr, mOutputOpNum,
				nullptr, 0, nullptr, mStatus);
			mbRun = true;

			if (TF_GetCode(mStatus) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

			//Free Memory
			delete[] imgData;
			TF_DeleteTensor(inputTensorArr[0]);
		}
	}

	else
	{
		if (bNormalize)
		{
			float* imgData = new float[imgNum * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				int currImgIdx = imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]) / float(255.);
						}
					}
				}
			}

			mInputDimsArr[0][0] = static_cast<long long>(imgNum);
			size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(imgNum);

			for (int iii = 0; iii < imgNum; ++iii)
			{
				cv::Mat tmp(640, 640, CV_32FC1);
				std::memcpy(tmp.data, imgData + iii * (cropSize.x * cropSize.y * nImageChannel), cropSize.x * cropSize.y * nImageChannel * sizeof(float));
				int a = 1;
			}

			TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
				mInputDimsArr[0],
				mInputDims[0],
				imgData,
				inputDataSize,
				dealloc,
				nullptr);

			inputTensorArr[0] = inputImgTensor;

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			{
				mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
				size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);
				outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
			}

			if (bReloadEveryRun && mbRun) ReloadModel();
			TF_SessionRun(mSession, mRunOptions,
				mInputOpsArr, inputTensorArr, mInputOpNum,
				mOutputOpsArr, outputTensorArr, mOutputOpNum,
				nullptr, 0, nullptr, mStatus);
			mbRun = true;

			if (TF_GetCode(mStatus) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

			//Free Memory
			delete[] imgData;
			TF_DeleteTensor(inputTensorArr[0]);
		}
		else
		{
			float* imgData = new float[imgNum * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
			for (int imgIdx = 0; imgIdx < imgNum; ++imgIdx)
			{
				int currImgIdx = imgIdx;
				int currXIdx = currImgIdx % itX;
				int currYIdx = currImgIdx / itX;

				int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
				int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
				if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
				if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

				for (int y = 0; y < cropSize.y; ++y)
				{
					for (int x = 0; x < cropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]);
						}
					}
				}
			}

			mInputDimsArr[0][0] = static_cast<long long>(imgNum);
			size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(imgNum);

			TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
				mInputDimsArr[0],
				mInputDims[0],
				imgData,
				inputDataSize,
				dealloc,
				nullptr);

			inputTensorArr[0] = inputImgTensor;

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			{
				mOutputDimsArr[opsIdx][0] = static_cast<long long>(imgNum);
				size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(imgNum);
				outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
			}

			if (bReloadEveryRun && mbRun) ReloadModel();
			TF_SessionRun(mSession, mRunOptions,
				mInputOpsArr, inputTensorArr, mInputOpNum,
				mOutputOpsArr, outputTensorArr, mOutputOpNum,
				nullptr, 0, nullptr, mStatus);
			mbRun = true;

			if (TF_GetCode(mStatus) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

			//Free Memory
			delete[] imgData;
			TF_DeleteTensor(inputTensorArr[0]);
		}
	}

	delete[] inputTensorArr;
	delete[] outputTensorArr;

	return true;
}

bool TFCore::Run(float*** inputImgArr, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor*[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor*[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}
	
	auto const dealloc = [](void*, std::size_t, void*) {};
	
	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
	{
		int currBatch = batch;
		if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
			currBatch = imgNum % batch;
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			float *imgData = new float[currBatch * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3]];
			if (bNormalize)
			{
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
						{
							imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = inputImgArr[batchIdx * batch + imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] / float(255.);
						}
					}
				}
			}
			else
			{
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
						{
							imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = inputImgArr[batchIdx * batch + imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx];
						}
					}
				}
			}
			mInputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
			size_t inputDataSize = mInputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);

			TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
				mInputDimsArr[opsIdx],
				mInputDims[opsIdx],
				imgData,
				inputDataSize,
				dealloc,
				nullptr);
			inputTensorArr[opsIdx] = inputImgTensor;

			delete[] imgData;
		}

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		{
			mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
			size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);

			outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
		}

		TF_SessionRun(mSession, mRunOptions,
			mInputOpsArr, inputTensorArr, mInputOpNum,
			mOutputOpsArr, outputTensorArr, mOutputOpNum,
			nullptr, 0, nullptr, mStatus);

		//Input Tensor 메모리 해제
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			TF_DeleteTensor(inputTensorArr[opsIdx]);
		}

		if (TF_GetCode(mStatus) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);
	}
	return true;
}

bool TFCore::Run(float** inputImgArr, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor*[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor*[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
	{
		int currBatch = batch;
		if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
			currBatch = imgNum % batch;

		float* imgData = new float[currBatch * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3]];
		if (bNormalize)
		{
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = inputImgArr[batchIdx * batch + imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx] / float(255.);
					}
				}
			}
		}
		else
		{
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = inputImgArr[batchIdx * batch + imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx];
					}
				}
			}
		}
		mInputDimsArr[0][0] = static_cast<long long>(currBatch);
		size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(currBatch);

		TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
			mInputDimsArr[0],
			mInputDims[0],
			imgData,
			inputDataSize,
			dealloc,
			nullptr);

		inputTensorArr[0] = inputImgTensor;

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		{
			mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
			size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);

			outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
		}

		TF_SessionRun(mSession, mRunOptions,
			mInputOpsArr, inputTensorArr, mInputOpNum,
			mOutputOpsArr, outputTensorArr, mOutputOpNum,
			nullptr, 0, nullptr, mStatus);

		//Input Tensor 메모리 해제
		TF_DeleteTensor(inputTensorArr[0]);

		if (TF_GetCode(mStatus) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);
	}
	return true;
}

bool TFCore::Run(unsigned char*** inputImgArr, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(unsigned char*));

	TF_Tensor** inputTensorArr = new TF_Tensor *[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor *[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
	{
		int currBatch = batch;
		if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
			currBatch = imgNum % batch;
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			float *imgData = new float[currBatch * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3]];
			if (bNormalize)
			{
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
						{
							imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = (float)(inputImgArr[batchIdx * batch + imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx]) / (float)(255.);
						}
					}
				}
			}
			else
			{
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					for (int pixIdx = 0; pixIdx < mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < mInputDimsArr[opsIdx][3]; ++chnIdx)
						{
							imgData[imgIdx * mInputDimsArr[opsIdx][1] * mInputDimsArr[opsIdx][2] * mInputDimsArr[opsIdx][3] + pixIdx * mInputDimsArr[opsIdx][3] + chnIdx] = (float)(inputImgArr[batchIdx * batch + imgIdx][opsIdx][pixIdx * mInputDimsArr[opsIdx][3] + chnIdx]);
						}
					}
				}
			}
			mInputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
			size_t inputDataSize = mInputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);

			TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
				mInputDimsArr[opsIdx],
				mInputDims[opsIdx],
				imgData,
				inputDataSize,
				dealloc,
				nullptr);
			inputTensorArr[opsIdx] = inputImgTensor;

			delete[] imgData;
		}

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		{
			mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
			size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);

			outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
		}

		TF_SessionRun(mSession, mRunOptions,
			mInputOpsArr, inputTensorArr, mInputOpNum,
			mOutputOpsArr, outputTensorArr, mOutputOpNum,
			nullptr, 0, nullptr, mStatus);

		//Input Tensor 메모리 해제
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			TF_DeleteTensor(inputTensorArr[opsIdx]);
		}

		if (TF_GetCode(mStatus) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);
	}
	return true;
}

bool TFCore::Run(unsigned char** inputImgArr, int batch, bool bNormalize)
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int imgNum = (int)(_msize(inputImgArr) / sizeof(float*));

	TF_Tensor** inputTensorArr = new TF_Tensor *[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor *[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
	{
		int currBatch = batch;
		if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
			currBatch = imgNum % batch;

		float* imgData = new float[currBatch * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3]];
		if (bNormalize)
		{
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = (float)(inputImgArr[batchIdx * batch + imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx]) / (float)(255.);
					}
				}
			}
		}
		else
		{
			for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
			{
				for (int pixIdx = 0; pixIdx < mInputDimsArr[0][1] * mInputDimsArr[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < mInputDimsArr[0][3]; ++chnIdx)
					{
						imgData[imgIdx * mInputDimsArr[0][1] * mInputDimsArr[0][2] * mInputDimsArr[0][3] + pixIdx * mInputDimsArr[0][3] + chnIdx] = (float)(inputImgArr[batchIdx * batch + imgIdx][pixIdx * mInputDimsArr[0][3] + chnIdx]);
					}
				}
			}
		}
		mInputDimsArr[0][0] = static_cast<long long>(currBatch);
		size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(currBatch);

		TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
			mInputDimsArr[0],
			mInputDims[0],
			imgData,
			inputDataSize,
			dealloc,
			nullptr);

		inputTensorArr[0] = inputImgTensor;

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		{
			mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
			size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);

			outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
		}

		TF_SessionRun(mSession, mRunOptions,
			mInputOpsArr, inputTensorArr, mInputOpNum,
			mOutputOpsArr, outputTensorArr, mOutputOpNum,
			nullptr, 0, nullptr, mStatus);

		//Input Tensor 메모리 해제
		TF_DeleteTensor(inputTensorArr[0]);

		if (TF_GetCode(mStatus) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
			mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);
	}
	return true;
}

bool TFCore::Run(unsigned char** inputImg, CPoint imgSize, CPoint cropSize, CPoint overlapSize, CPoint buffPos, int batch, bool bNormalize, bool bConvertGrayToColor, bool bReloadEveryRun)
//VisionWorks image input format, has only one input operator
{
	if (!mIsModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	if ((cropSize.x <= overlapSize.x) || (cropSize.y <= overlapSize.y))
	{
		std::cout << "Crop size must be larger than overlap size." << std::endl;
		return false;
	}

	TF_Tensor** inputTensorArr = new TF_Tensor*[mInputOpNum];
	TF_Tensor** outputTensorArr = new TF_Tensor*[mOutputOpNum];

	if (!(mOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
			}
			mOutputTensors[opsIdx].clear();
		}
		mOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < mOutputOpNum; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		mOutputTensors.push_back(vt);
	}

	auto const dealloc = [](void*, std::size_t, void*) {};

	mImageSize = imgSize;
	mCropSize = cropSize;
	mOverlapSize = overlapSize;

	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - (mCropSize.x - mOverlapSize.x) * (itX - 1) > mCropSize.x) ++itX;
	if (mImageSize.y - (mCropSize.y - mOverlapSize.y) * (itY - 1) > mCropSize.y) ++itY;

	int imgNum = itX * itY;
	int nImageChannel = (int)(mInputDimsArr[0][3]);
	int batchIterNum = imgNum / batch + (int)(bool)(imgNum % batch);

	if (bConvertGrayToColor)
	{
		if (bNormalize)
		{
			for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
			{
				int currBatch = batch;
				if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
					currBatch = imgNum % batch;

				float* imgData = new float[currBatch * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					int currImgIdx = batchIdx * batch + imgIdx;
					int currXIdx = currImgIdx % itX;
					int currYIdx = currImgIdx / itX;

					int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
					int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
					if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
					if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

					for (int y = 0; y < cropSize.y; ++y)
					{
						for (int x = 0; x < cropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]) / float(255.);
							}
						}
					}
				}

				mInputDimsArr[0][0] = static_cast<long long>(currBatch);
				size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(currBatch);

				TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
					mInputDimsArr[0],
					mInputDims[0],
					imgData,
					inputDataSize,
					dealloc,
					nullptr);

				inputTensorArr[0] = inputImgTensor;

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				{
					mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
					size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);
					outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
				}

				if (bReloadEveryRun && mbRun) ReloadModel();
				TF_SessionRun(mSession, mRunOptions,
					mInputOpsArr, inputTensorArr, mInputOpNum,
					mOutputOpsArr, outputTensorArr, mOutputOpNum,
					nullptr, 0, nullptr, mStatus);
				mbRun = true;

				if (TF_GetCode(mStatus) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
					mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

				//Free Memory
				delete[] imgData;
				TF_DeleteTensor(inputTensorArr[0]);
			}
		}
		else
		{
			for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
			{
				int currBatch = batch;
				if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
					currBatch = imgNum % batch;

				float* imgData = new float[currBatch * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					int currImgIdx = batchIdx * batch + imgIdx;
					int currXIdx = currImgIdx % itX;
					int currYIdx = currImgIdx / itX;

					int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
					int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
					if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
					if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

					for (int y = 0; y < cropSize.y; ++y)
					{
						for (int x = 0; x < cropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x]);
							}
						}
					}
				}

				mInputDimsArr[0][0] = static_cast<long long>(currBatch);
				size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(currBatch);

				TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
					mInputDimsArr[0],
					mInputDims[0],
					imgData,
					inputDataSize,
					dealloc,
					nullptr);

				inputTensorArr[0] = inputImgTensor;

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				{
					mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
					size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);
					outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
				}

				if (bReloadEveryRun && mbRun) ReloadModel();
				TF_SessionRun(mSession, mRunOptions,
					mInputOpsArr, inputTensorArr, mInputOpNum,
					mOutputOpsArr, outputTensorArr, mOutputOpNum,
					nullptr, 0, nullptr, mStatus);
				mbRun = true;

				if (TF_GetCode(mStatus) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
					mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

				//Free Memory
				delete[] imgData;
				TF_DeleteTensor(inputTensorArr[0]);
			}
		}
	}
	else
	{
		if (bNormalize)
		{
			for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
			{
				int currBatch = batch;
				if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
					currBatch = imgNum % batch;

				float* imgData = new float[currBatch * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					int currImgIdx = batchIdx * batch + imgIdx;
					int currXIdx = currImgIdx % itX;
					int currYIdx = currImgIdx / itX;

					int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
					int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
					if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
					if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

					for (int y = 0; y < cropSize.y; ++y)
					{
						for (int x = 0; x < cropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]) / float(255.);
							}
						}
					}
				}

				mInputDimsArr[0][0] = static_cast<long long>(currBatch);
				size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(currBatch);

				TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
					mInputDimsArr[0],
					mInputDims[0],
					imgData,
					inputDataSize,
					dealloc,
					nullptr);

				inputTensorArr[0] = inputImgTensor;

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				{
					mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
					size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);
					outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
				}

				if (bReloadEveryRun && mbRun) ReloadModel();
				TF_SessionRun(mSession, mRunOptions,
					mInputOpsArr, inputTensorArr, mInputOpNum,
					mOutputOpsArr, outputTensorArr, mOutputOpNum,
					nullptr, 0, nullptr, mStatus);
				mbRun = true;

				if (TF_GetCode(mStatus) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
					mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

				//Free Memory
				delete[] imgData;
				TF_DeleteTensor(inputTensorArr[0]);
			}
		}
		else
		{
			for (int batchIdx = 0; batchIdx < batchIterNum; ++batchIdx)
			{
				int currBatch = batch;
				if ((batchIdx == batchIterNum - 1) && (imgNum % batch != 0))
					currBatch = imgNum % batch;

				float* imgData = new float[currBatch * mInputDimsArr[0][1] * mInputDimsArr[0][2] * nImageChannel];
				for (int imgIdx = 0; imgIdx < currBatch; ++imgIdx)
				{
					int currImgIdx = batchIdx * batch + imgIdx;
					int currXIdx = currImgIdx % itX;
					int currYIdx = currImgIdx / itX;

					int currX = (mCropSize.x - mOverlapSize.x) * currXIdx;
					int currY = (mCropSize.y - mOverlapSize.y) * currYIdx;
					if (currX + mCropSize.x > mImageSize.x) currX = mImageSize.x - mCropSize.x;
					if (currY + mCropSize.y > mImageSize.y) currY = mImageSize.y - mCropSize.y;

					for (int y = 0; y < cropSize.y; ++y)
					{
						for (int x = 0; x < cropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								imgData[imgIdx * cropSize.y * cropSize.x * nImageChannel + y * cropSize.x * nImageChannel + x * nImageChannel + c] = float(inputImg[buffPos.y + currY + y][buffPos.x + currX + x * nImageChannel + c]);
							}
						}
					}
				}

				mInputDimsArr[0][0] = static_cast<long long>(currBatch);
				size_t inputDataSize = mInputDataSizePerBatch[0] * static_cast<size_t>(currBatch);

				TF_Tensor* inputImgTensor = TF_NewTensor(TF_FLOAT,
					mInputDimsArr[0],
					mInputDims[0],
					imgData,
					inputDataSize,
					dealloc,
					nullptr);

				inputTensorArr[0] = inputImgTensor;

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
				{
					mOutputDimsArr[opsIdx][0] = static_cast<long long>(currBatch);
					size_t outputDataSize = mOutputDataSizePerBatch[opsIdx] * static_cast<size_t>(currBatch);
					outputTensorArr[opsIdx] = TF_AllocateTensor(TF_FLOAT, mOutputDimsArr[opsIdx], mOutputDims[opsIdx], outputDataSize);
				}

				if (bReloadEveryRun && mbRun) ReloadModel();
				TF_SessionRun(mSession, mRunOptions,
					mInputOpsArr, inputTensorArr, mInputOpNum,
					mOutputOpsArr, outputTensorArr, mOutputOpNum,
					nullptr, 0, nullptr, mStatus);
				mbRun = true;

				if (TF_GetCode(mStatus) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
					mOutputTensors[opsIdx].push_back(outputTensorArr[opsIdx]);

				//Free Memory
				delete[] imgData;
				TF_DeleteTensor(inputTensorArr[0]);
			}
		}
	}

	delete[] inputTensorArr;
	delete[] outputTensorArr;

	return true;
}

bool TFCore::FreeModel()
{
	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
		delete[] mInputDimsArr[opsIdx];
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
		delete[] mOutputDimsArr[opsIdx];
	delete[] mInputOpsArr;
	delete[] mOutputOpsArr;
	delete[] mInputDims;
	delete[] mOutputDims;
	delete[] mInputDataSizePerBatch;
	delete[] mOutputDataSizePerBatch;
	delete[] mInputDimsArr;
	delete[] mOutputDimsArr;

	if (mRunOptions != nullptr)
		TF_DeleteBuffer(mRunOptions);
	if (mMetaGraph != nullptr)
		TF_DeleteBuffer(mMetaGraph);
	if (mSession != nullptr)
	{
		TF_CloseSession(mSession, mStatus);
		TF_DeleteSession(mSession, mStatus);
	}
	if (mSessionOptions != nullptr)
		TF_DeleteSessionOptions(mSessionOptions);
	if (mGraph != nullptr)
		TF_DeleteGraph(mGraph);
	if (mStatus != nullptr)
		TF_DeleteStatus(mStatus);

	mIsModelLoaded = false;
	mIsDataLoaded = false;

	return true;
}

bool TFCore::IsModelLoaded()
{
	return mIsModelLoaded;
}

long long** TFCore::GetInputDims()
{
	return mInputDimsArr;
}

long long** TFCore::GetOutputDims()
{
	return mOutputDimsArr;
}

bool TFCore::_Run()
{
	return true;
}