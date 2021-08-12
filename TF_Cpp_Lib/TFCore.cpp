#pragma once
#include "TFCore.h"


TFCore::TFCore()
{
	m_Session = nullptr;
	m_RunOptions = nullptr;
	m_SessionOptions = nullptr;
	m_Graph = nullptr;
	m_Status = nullptr;
	m_MetaGraph = nullptr;
	m_arrInputOps = nullptr;
	m_arrOutputOps = nullptr;
	m_nInputDims = nullptr;
	m_nOutputDims = nullptr;
	m_InputDims = nullptr;
	m_OutputDims = nullptr;
	m_InputDataSizePerBatch = nullptr;
	m_OutputDataSizePerBatch = nullptr;

	m_bModelLoaded = false;
	m_bDataLoaded = false;
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

bool TFCore::LoadModel(const char* ModelPath, std::vector<const char*> &vtInputOpNames, std::vector<const char*>& vtOutputOpNames)
{
	m_ModelPath = ModelPath;
	m_RunOptions = TF_NewBufferFromString("", 0);
	//m_SessionOptions = CreateSessionOptions(0.99);
	m_SessionOptions = TF_NewSessionOptions();
	m_Graph = TF_NewGraph();
	m_Status = TF_NewStatus();
	m_MetaGraph = TF_NewBuffer();
	m_Session = TF_NewSession(m_Graph, m_SessionOptions, m_Status);
	const char* tag = "serve";
	m_Session = TF_LoadSessionFromSavedModel(m_SessionOptions, m_RunOptions, m_ModelPath, &tag, 1, m_Graph, m_MetaGraph, m_Status);

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
		char InputOpFullName[200];
		strcpy_s(InputOpFullName, sizeof(InputOpFullName), (char*)vtInputOpNames[i]);
		char* chIndex = NULL;
		const char* InputOpName = strtok_s(InputOpFullName, ":", &chIndex);
		int nInputOpOutputIndex = atoi(chIndex);
		TF_Operation* InputOp = TF_GraphOperationByName(m_Graph, InputOpName);
		if (InputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		m_arrInputOps[i] = TF_Output{ InputOp, nInputOpOutputIndex };
		m_nInputDims[i] = TF_GraphGetTensorNumDims(m_Graph, m_arrInputOps[i], m_Status);
		int64_t* InputShape = new int64_t[m_nInputDims[i]];
		TF_GraphGetTensorShape(m_Graph, m_arrInputOps[i], InputShape, m_nInputDims[i], m_Status);
		m_InputDims[i] = new long long[m_nInputDims[i]];
		m_InputDims[i][0] = static_cast<long long>(1);
		for (int j = 1; j < m_nInputDims[i]; ++j) m_InputDims[i][j] = static_cast<long long>(InputShape[j]);
		m_InputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(m_arrInputOps[i]));
		for (int j = 1; j < m_nInputDims[i]; ++j) m_InputDataSizePerBatch[i] = m_InputDataSizePerBatch[i] * static_cast<int>(InputShape[j]);
		delete[] InputShape;
	}

	for (int i = 0; i < m_nOutputOps; ++i)
	{
		char OutputOpFullName[200];
		strcpy_s(OutputOpFullName, sizeof(OutputOpFullName), (char*)vtOutputOpNames[i]);
		char* chIndex = NULL;
		const char* OutputOpName = strtok_s(OutputOpFullName, ":", &chIndex);
		int nOutputOpOutputIndex = atoi(chIndex);
		TF_Operation* OutputOp = TF_GraphOperationByName(m_Graph, OutputOpName);
		if (OutputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		int n = TF_OperationNumOutputs(OutputOp);
		m_arrOutputOps[i] = TF_Output{ OutputOp, nOutputOpOutputIndex };
		m_nOutputDims[i] = TF_GraphGetTensorNumDims(m_Graph, m_arrOutputOps[i], m_Status);
		int64_t* OutputShape = new int64_t[m_nOutputDims[i]];
		TF_GraphGetTensorShape(m_Graph, m_arrOutputOps[i], OutputShape, m_nOutputDims[i], m_Status);
		m_OutputDims[i] = new long long[m_nOutputDims[i]];
		m_OutputDims[i][0] = static_cast<long long>(1);
		for (int j = 1; j < m_nOutputDims[i]; ++j) m_OutputDims[i][j] = static_cast<long long>(OutputShape[j]);
		m_OutputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(m_arrOutputOps[i]));
		for (int j = 1; j < m_nOutputDims[i]; ++j) m_OutputDataSizePerBatch[i] = m_OutputDataSizePerBatch[i] * static_cast<int>(OutputShape[j]);
		delete[] OutputShape;
	}
	m_bModelLoaded = true;
	return true;
}

bool TFCore::Run(float*** pImageSet, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}

	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
	{
		float* ImageData = new float[nImage * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]];
		if (bNormalize)
		{
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
				{
					ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] + pixIdx] = pImageSet[dataIdx][opsIdx][pixIdx] / float(255.);
				}
			}
		}
		else
		{
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
				{
					ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] + pixIdx] = pImageSet[dataIdx][opsIdx][pixIdx];
				}
			}
		}
		m_InputDims[opsIdx][0] = static_cast<long long>(nImage);
		size_t InputDataSize = m_InputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);

		TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
			m_InputDims[opsIdx],
			m_nInputDims[opsIdx],
			ImageData,
			InputDataSize,
			Deallocator,
			nullptr);
		arrInputTensors[opsIdx] = InputImageTensor;

		delete[] ImageData;
	}

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
	{
		m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
		size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);
		arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
	}

	TF_SessionRun(m_Session, m_RunOptions,
		m_arrInputOps, arrInputTensors, m_nInputOps,
		m_arrOutputOps, arrOutputTensors, m_nOutputOps,
		nullptr, 0, nullptr, m_Status);

	//Input Tensor 메모리 해제
	for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
	{
		TF_DeleteTensor(arrInputTensors[opsIdx]);
	}

	if (TF_GetCode(m_Status) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

	return true;
}

bool TFCore::Run(float** pImageSet, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	float* ImageData = new float[nImage * m_InputDims[0][1] * m_InputDims[0][2]];
	if (bNormalize)
	{
		for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
		{
			for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
			{
				ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] + pixIdx] = pImageSet[dataIdx][pixIdx] / float(255.);
			}
		}
	}
	else
	{
		for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
		{
			for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
			{
				ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] + pixIdx] = pImageSet[dataIdx][pixIdx];
			}
		}
	}
	m_InputDims[0][0] = static_cast<long long>(nImage);
	size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nImage);

	TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
		m_InputDims[0],
		m_nInputDims[0],
		ImageData,
		InputDataSize,
		Deallocator,
		nullptr);

	arrInputTensors[0] = InputImageTensor;

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
	{
		m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
		size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);

		arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
	}

	TF_SessionRun(m_Session, m_RunOptions,
		m_arrInputOps, arrInputTensors, m_nInputOps,
		m_arrOutputOps, arrOutputTensors, m_nOutputOps,
		nullptr, 0, nullptr, m_Status);

	//Input Tensor 메모리 해제
	TF_DeleteTensor(arrInputTensors[0]);

	if (TF_GetCode(m_Status) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

	return true;
}

bool TFCore::Run(unsigned char*** pImageSet, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor *[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor *[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
	{
		float* ImageData = new float[nImage * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]];
		if (bNormalize)
		{
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
				{
					ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] + pixIdx] = pImageSet[dataIdx][opsIdx][pixIdx] / float(255.);
				}
			}
		}
		else
		{
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
				{
					ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] + pixIdx] = pImageSet[dataIdx][opsIdx][pixIdx];
				}
			}
		}
		m_InputDims[opsIdx][0] = static_cast<long long>(nImage);
		size_t InputDataSize = m_InputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);

		TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
			m_InputDims[opsIdx],
			m_nInputDims[opsIdx],
			ImageData,
			InputDataSize,
			Deallocator,
			nullptr);
		arrInputTensors[opsIdx] = InputImageTensor;

		delete[] ImageData;
	}

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
	{
		m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
		size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);

		arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
	}

	TF_SessionRun(m_Session, m_RunOptions,
		m_arrInputOps, arrInputTensors, m_nInputOps,
		m_arrOutputOps, arrOutputTensors, m_nOutputOps,
		nullptr, 0, nullptr, m_Status);

	//Input Tensor 메모리 해제
	for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
	{
		TF_DeleteTensor(arrInputTensors[opsIdx]);
	}

	if (TF_GetCode(m_Status) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

	return true;
}

bool TFCore::Run(unsigned char** pImageSet, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor *[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor *[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	float* ImageData = new float[nImage * m_InputDims[0][1] * m_InputDims[0][2]];
	if (bNormalize)
	{
		for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
		{
			for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
			{
				ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] + pixIdx] = (float)(pImageSet[dataIdx][pixIdx]) / (float)(255.);
			}
		}
	}
	else
	{
		for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
		{
			for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
			{
				ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] + pixIdx] = (float)(pImageSet[dataIdx][pixIdx]);
			}
		}
	}
	m_InputDims[0][0] = static_cast<long long>(nImage);
	size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nImage);

	TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
		m_InputDims[0],
		m_nInputDims[0],
		ImageData,
		InputDataSize,
		Deallocator,
		nullptr);

	arrInputTensors[0] = InputImageTensor;

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
	{
		m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
		size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);

		arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
	}

	TF_SessionRun(m_Session, m_RunOptions,
		m_arrInputOps, arrInputTensors, m_nInputOps,
		m_arrOutputOps, arrOutputTensors, m_nOutputOps,
		nullptr, 0, nullptr, m_Status);

	//Input Tensor 메모리 해제
	TF_DeleteTensor(arrInputTensors[0]);

	if (TF_GetCode(m_Status) != TF_OK)
	{
		return false;
	}

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

	return true;
}

bool TFCore::Run(unsigned char** ppImage, CPoint ptImageSize, CPoint ptCropSize, CPoint ptOverlapSize, CPoint ptBuffPos, bool bNormalize, bool bConvertGrayToColor)
//VisionWorks image input format, has only one input operator
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	if ((ptCropSize.x <= ptOverlapSize.x) || (ptCropSize.y <= ptOverlapSize.y))
	{
		std::cout << "Crop size must be larger than overlap size." << std::endl;
		return false;
	}

	TF_Tensor** arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	m_ptImageSize = ptImageSize;
	m_ptCropSize = ptCropSize;
	m_ptOverlapSize = ptOverlapSize;

	int nIterX = (int)(m_ptImageSize.x / (m_ptCropSize.x - m_ptOverlapSize.x));
	int nIterY = (int)(m_ptImageSize.y / (m_ptCropSize.y - m_ptOverlapSize.y));
	if (m_ptImageSize.x - ((m_ptCropSize.x - m_ptOverlapSize.x) * (nIterX - 1)) > m_ptCropSize.x) ++nIterX;
	if (m_ptImageSize.y - ((m_ptCropSize.y - m_ptOverlapSize.y) * (nIterY - 1)) > m_ptCropSize.y) ++nIterY;

	int nImage = nIterX * nIterY;
	int nImageChannel = (int)(m_InputDims[0][3]);

	if (bConvertGrayToColor)
	{
		if (bNormalize)
		{
			float* ImageData = new float[nImage * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				int nCurrImgIdx = dataIdx;
				int nCurrXIdx = nCurrImgIdx % nIterX;
				int nCurrYIdx = nCurrImgIdx / nIterX;

				int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
				int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
				if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
				if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

				for (int y = 0; y < ptCropSize.y; ++y)
				{
					for (int x = 0; x < ptCropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x]) / float(255.);
						}
					}
				}
			}

			m_InputDims[0][0] = static_cast<long long>(nImage);
			size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nImage);

			TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
				m_InputDims[0],
				m_nInputDims[0],
				ImageData,
				InputDataSize,
				Deallocator,
				nullptr);

			arrInputTensors[0] = InputImageTensor;

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			{
				m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
				size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);
				arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
			}

			TF_SessionRun(m_Session, m_RunOptions,
				m_arrInputOps, arrInputTensors, m_nInputOps,
				m_arrOutputOps, arrOutputTensors, m_nOutputOps,
				nullptr, 0, nullptr, m_Status);

			if (TF_GetCode(m_Status) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

			//Free Memory
			delete[] ImageData;
			TF_DeleteTensor(arrInputTensors[0]);
			delete[] arrInputTensors;
			delete[] arrOutputTensors;
		}
		else
		{
			float* ImageData = new float[nImage * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				int nCurrImgIdx = dataIdx;
				int nCurrXIdx = nCurrImgIdx % nIterX;
				int nCurrYIdx = nCurrImgIdx / nIterX;

				int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
				int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
				if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
				if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

				for (int y = 0; y < ptCropSize.y; ++y)
				{
					for (int x = 0; x < ptCropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x]);
						}
					}
				}
			}

			m_InputDims[0][0] = static_cast<long long>(nImage);
			size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nImage);

			TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
				m_InputDims[0],
				m_nInputDims[0],
				ImageData,
				InputDataSize,
				Deallocator,
				nullptr);

			arrInputTensors[0] = InputImageTensor;

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			{
				m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
				size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);
				arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
			}

			TF_SessionRun(m_Session, m_RunOptions,
				m_arrInputOps, arrInputTensors, m_nInputOps,
				m_arrOutputOps, arrOutputTensors, m_nOutputOps,
				nullptr, 0, nullptr, m_Status);

			if (TF_GetCode(m_Status) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

			//Free Memory
			delete[] ImageData;
			TF_DeleteTensor(arrInputTensors[0]);
			delete[] arrInputTensors;
			delete[] arrOutputTensors;
		}
	}

	else
	{
		if (bNormalize)
		{
			float* ImageData = new float[nImage * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				int nCurrImgIdx = dataIdx;
				int nCurrXIdx = nCurrImgIdx % nIterX;
				int nCurrYIdx = nCurrImgIdx / nIterX;

				int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
				int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
				if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
				if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

				for (int y = 0; y < ptCropSize.y; ++y)
				{
					for (int x = 0; x < ptCropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x * nImageChannel + c]) / float(255.);
						}
					}
				}
			}

			m_InputDims[0][0] = static_cast<long long>(nImage);
			size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nImage);

			for (int iii = 0; iii < nImage; ++iii)
			{
				cv::Mat tmp(640, 640, CV_32FC1);
				std::memcpy(tmp.data, ImageData + iii * (ptCropSize.x * ptCropSize.y * nImageChannel), ptCropSize.x * ptCropSize.y * nImageChannel * sizeof(float));
				int a = 1;
			}

			TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
				m_InputDims[0],
				m_nInputDims[0],
				ImageData,
				InputDataSize,
				Deallocator,
				nullptr);

			arrInputTensors[0] = InputImageTensor;

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			{
				m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
				size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);
				arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
			}

			TF_SessionRun(m_Session, m_RunOptions,
				m_arrInputOps, arrInputTensors, m_nInputOps,
				m_arrOutputOps, arrOutputTensors, m_nOutputOps,
				nullptr, 0, nullptr, m_Status);

			if (TF_GetCode(m_Status) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

			//Free Memory
			delete[] ImageData;
			TF_DeleteTensor(arrInputTensors[0]);
			delete[] arrInputTensors;
			delete[] arrOutputTensors;
		}
		else
		{
			float* ImageData = new float[nImage * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
			for (int dataIdx = 0; dataIdx < nImage; ++dataIdx)
			{
				int nCurrImgIdx = dataIdx;
				int nCurrXIdx = nCurrImgIdx % nIterX;
				int nCurrYIdx = nCurrImgIdx / nIterX;

				int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
				int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
				if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
				if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

				for (int y = 0; y < ptCropSize.y; ++y)
				{
					for (int x = 0; x < ptCropSize.x; ++x)
					{
						for (int c = 0; c < nImageChannel; ++c)
						{
							ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x * nImageChannel + c]);
						}
					}
				}
			}

			m_InputDims[0][0] = static_cast<long long>(nImage);
			size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nImage);

			TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
				m_InputDims[0],
				m_nInputDims[0],
				ImageData,
				InputDataSize,
				Deallocator,
				nullptr);

			arrInputTensors[0] = InputImageTensor;

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			{
				m_OutputDims[opsIdx][0] = static_cast<long long>(nImage);
				size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nImage);
				arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
			}

			TF_SessionRun(m_Session, m_RunOptions,
				m_arrInputOps, arrInputTensors, m_nInputOps,
				m_arrOutputOps, arrOutputTensors, m_nOutputOps,
				nullptr, 0, nullptr, m_Status);

			if (TF_GetCode(m_Status) != TF_OK)
			{
				return false;
			}

			for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);

			//Free Memory
			delete[] ImageData;
			TF_DeleteTensor(arrInputTensors[0]);
			delete[] arrInputTensors;
			delete[] arrOutputTensors;
		}
	}

	return true;
}

bool TFCore::Run(float*** pImageSet, int nBatch, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
	{
		int nCurrBatch = nBatch;
		if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
			nCurrBatch = nImage % nBatch;
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			float *ImageData = new float[nCurrBatch * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] * m_InputDims[opsIdx][3]];
			if (bNormalize)
			{
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < m_InputDims[opsIdx][3]; ++chnIdx)
						{
							ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] * m_InputDims[opsIdx][3] + pixIdx * m_InputDims[opsIdx][3] + chnIdx] = pImageSet[batchIdx * nBatch + dataIdx][opsIdx][pixIdx * m_InputDims[opsIdx][3] + chnIdx] / float(255.);
						}
					}
				}
			}
			else
			{
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < m_InputDims[opsIdx][3]; ++chnIdx)
						{
							ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] * m_InputDims[opsIdx][3] + pixIdx * m_InputDims[opsIdx][3] + chnIdx] = pImageSet[batchIdx * nBatch + dataIdx][opsIdx][pixIdx * m_InputDims[opsIdx][3] + chnIdx];
						}
					}
				}
			}
			m_InputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t InputDataSize = m_InputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
				m_InputDims[opsIdx],
				m_nInputDims[opsIdx],
				ImageData,
				InputDataSize,
				Deallocator,
				nullptr);
			arrInputTensors[opsIdx] = InputImageTensor;

			delete[] ImageData;
		}

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
		}

		TF_SessionRun(m_Session, m_RunOptions,
			m_arrInputOps, arrInputTensors, m_nInputOps,
			m_arrOutputOps, arrOutputTensors, m_nOutputOps,
			nullptr, 0, nullptr, m_Status);

		//Input Tensor 메모리 해제
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			TF_DeleteTensor(arrInputTensors[opsIdx]);
		}

		if (TF_GetCode(m_Status) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
	}
	return true;
}

bool TFCore::Run(float** pImageSet, int nBatch, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
	{
		int nCurrBatch = nBatch;
		if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
			nCurrBatch = nImage % nBatch;

		float* ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2] * m_InputDims[0][3]];
		if (bNormalize)
		{
			for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < m_InputDims[0][3]; ++chnIdx)
					{
						ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] * m_InputDims[0][3] + pixIdx * m_InputDims[0][3] + chnIdx] = pImageSet[batchIdx * nBatch + dataIdx][pixIdx * m_InputDims[0][3] + chnIdx] / float(255.);
					}
				}
			}
		}
		else
		{
			for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < m_InputDims[0][3]; ++chnIdx)
					{
						ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] * m_InputDims[0][3] + pixIdx * m_InputDims[0][3] + chnIdx] = pImageSet[batchIdx * nBatch + dataIdx][pixIdx * m_InputDims[0][3] + chnIdx];
					}
				}
			}
		}
		m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
		size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

		TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
			m_InputDims[0],
			m_nInputDims[0],
			ImageData,
			InputDataSize,
			Deallocator,
			nullptr);

		arrInputTensors[0] = InputImageTensor;

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
		}

		TF_SessionRun(m_Session, m_RunOptions,
			m_arrInputOps, arrInputTensors, m_nInputOps,
			m_arrOutputOps, arrOutputTensors, m_nOutputOps,
			nullptr, 0, nullptr, m_Status);

		//Input Tensor 메모리 해제
		TF_DeleteTensor(arrInputTensors[0]);

		if (TF_GetCode(m_Status) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
	}
	return true;
}

bool TFCore::Run(unsigned char*** pImageSet, int nBatch, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(unsigned char*));

	TF_Tensor** arrInputTensors = new TF_Tensor *[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor *[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
	{
		int nCurrBatch = nBatch;
		if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
			nCurrBatch = nImage % nBatch;
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			float *ImageData = new float[nCurrBatch * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] * m_InputDims[opsIdx][3]];
			if (bNormalize)
			{
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < m_InputDims[opsIdx][3]; ++chnIdx)
						{
							ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] * m_InputDims[opsIdx][3] + pixIdx * m_InputDims[opsIdx][3] + chnIdx] = (float)(pImageSet[batchIdx * nBatch + dataIdx][opsIdx][pixIdx * m_InputDims[opsIdx][3] + chnIdx]) / (float)(255.);
						}
					}
				}
			}
			else
			{
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
					{
						for (int chnIdx = 0; chnIdx < m_InputDims[opsIdx][3]; ++chnIdx)
						{
							ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] * m_InputDims[opsIdx][3] + pixIdx * m_InputDims[opsIdx][3] + chnIdx] = (float)(pImageSet[batchIdx * nBatch + dataIdx][opsIdx][pixIdx * m_InputDims[opsIdx][3] + chnIdx]);
						}
					}
				}
			}
			m_InputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t InputDataSize = m_InputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
				m_InputDims[opsIdx],
				m_nInputDims[opsIdx],
				ImageData,
				InputDataSize,
				Deallocator,
				nullptr);
			arrInputTensors[opsIdx] = InputImageTensor;

			delete[] ImageData;
		}

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
		}

		TF_SessionRun(m_Session, m_RunOptions,
			m_arrInputOps, arrInputTensors, m_nInputOps,
			m_arrOutputOps, arrOutputTensors, m_nOutputOps,
			nullptr, 0, nullptr, m_Status);

		//Input Tensor 메모리 해제
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			TF_DeleteTensor(arrInputTensors[opsIdx]);
		}

		if (TF_GetCode(m_Status) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
	}
	return true;
}

bool TFCore::Run(unsigned char** pImageSet, int nBatch, bool bNormalize)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	int nImage = (int)(_msize(pImageSet) / sizeof(float*));

	TF_Tensor** arrInputTensors = new TF_Tensor *[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor *[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
	{
		int nCurrBatch = nBatch;
		if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
			nCurrBatch = nImage % nBatch;

		float* ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2] * m_InputDims[0][3]];
		if (bNormalize)
		{
			for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < m_InputDims[0][3]; ++chnIdx)
					{
						ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] * m_InputDims[0][3] + pixIdx * m_InputDims[0][3] + chnIdx] = (float)(pImageSet[batchIdx * nBatch + dataIdx][pixIdx * m_InputDims[0][3] + chnIdx]) / (float)(255.);
					}
				}
			}
		}
		else
		{
			for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
				{
					for (int chnIdx = 0; chnIdx < m_InputDims[0][3]; ++chnIdx)
					{
						ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] * m_InputDims[0][3] + pixIdx * m_InputDims[0][3] + chnIdx] = (float)(pImageSet[batchIdx * nBatch + dataIdx][pixIdx * m_InputDims[0][3] + chnIdx]);
					}
				}
			}
		}
		m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
		size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

		TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
			m_InputDims[0],
			m_nInputDims[0],
			ImageData,
			InputDataSize,
			Deallocator,
			nullptr);

		arrInputTensors[0] = InputImageTensor;

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		{
			m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
		}

		TF_SessionRun(m_Session, m_RunOptions,
			m_arrInputOps, arrInputTensors, m_nInputOps,
			m_arrOutputOps, arrOutputTensors, m_nOutputOps,
			nullptr, 0, nullptr, m_Status);

		//Input Tensor 메모리 해제
		TF_DeleteTensor(arrInputTensors[0]);

		if (TF_GetCode(m_Status) != TF_OK)
		{
			return false;
		}

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
	}
	return true;
}

bool TFCore::Run(unsigned char** ppImage, CPoint ptImageSize, CPoint ptCropSize, CPoint ptOverlapSize, CPoint ptBuffPos, int nBatch, bool bNormalize, bool bConvertGrayToColor)
//VisionWorks image input format, has only one input operator
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	if ((ptCropSize.x <= ptOverlapSize.x) || (ptCropSize.y <= ptOverlapSize.y))
	{
		std::cout << "Crop size must be larger than overlap size." << std::endl;
		return false;
	}

	TF_Tensor** arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor** arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(m_vtOutputTensors.empty()))
	{
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
			{
				TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
			}
			m_vtOutputTensors[opsIdx].clear();
		}
		m_vtOutputTensors.clear();
	}
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor*> vt;
		m_vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void*, std::size_t, void*) {};

	m_ptImageSize = ptImageSize;
	m_ptCropSize = ptCropSize;
	m_ptOverlapSize = ptOverlapSize;

	int nIterX = (int)(m_ptImageSize.x / (m_ptCropSize.x - m_ptOverlapSize.x));
	int nIterY = (int)(m_ptImageSize.y / (m_ptCropSize.y - m_ptOverlapSize.y));
	if (m_ptImageSize.x - ((m_ptCropSize.x - m_ptOverlapSize.x) * (nIterX - 1)) > m_ptCropSize.x) ++nIterX;
	if (m_ptImageSize.y - ((m_ptCropSize.y - m_ptOverlapSize.y) * (nIterY - 1)) > m_ptCropSize.y) ++nIterY;
	
	int nImage = nIterX * nIterY;
	int nImageChannel = (int)(m_InputDims[0][3]);
	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	if (bConvertGrayToColor)
	{
		if (bNormalize)
		{
			for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
			{
				int nCurrBatch = nBatch;
				if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
					nCurrBatch = nImage % nBatch;

				float* ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					int nCurrImgIdx = batchIdx * nBatch + dataIdx;
					int nCurrXIdx = nCurrImgIdx % nIterX;
					int nCurrYIdx = nCurrImgIdx / nIterX;

					int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
					int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
					if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
					if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

					for (int y = 0; y < ptCropSize.y; ++y)
					{
						for (int x = 0; x < ptCropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x]) / float(255.);
							}
						}
					}
				}

				m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
				size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

				TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
					m_InputDims[0],
					m_nInputDims[0],
					ImageData,
					InputDataSize,
					Deallocator,
					nullptr);

				arrInputTensors[0] = InputImageTensor;

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				{
					m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
					size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);
					arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
				}

				TF_SessionRun(m_Session, m_RunOptions,
					m_arrInputOps, arrInputTensors, m_nInputOps,
					m_arrOutputOps, arrOutputTensors, m_nOutputOps,
					nullptr, 0, nullptr, m_Status);

				//Input Tensor 메모리 해제
				TF_DeleteTensor(arrInputTensors[0]);

				delete[] ImageData;
				if (TF_GetCode(m_Status) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
					m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
			}
		}
		else
		{
			for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
			{
				int nCurrBatch = nBatch;
				if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
					nCurrBatch = nImage % nBatch;

				float* ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					int nCurrImgIdx = batchIdx * nBatch + dataIdx;
					int nCurrXIdx = nCurrImgIdx % nIterX;
					int nCurrYIdx = nCurrImgIdx / nIterX;

					int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
					int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
					if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
					if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

					for (int y = 0; y < ptCropSize.y; ++y)
					{
						for (int x = 0; x < ptCropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x]);
							}
						}
					}
				}

				m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
				size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

				TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
					m_InputDims[0],
					m_nInputDims[0],
					ImageData,
					InputDataSize,
					Deallocator,
					nullptr);

				arrInputTensors[0] = InputImageTensor;

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				{
					m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
					size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);
					arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
				}

				TF_SessionRun(m_Session, m_RunOptions,
					m_arrInputOps, arrInputTensors, m_nInputOps,
					m_arrOutputOps, arrOutputTensors, m_nOutputOps,
					nullptr, 0, nullptr, m_Status);

				//Input Tensor 메모리 해제
				TF_DeleteTensor(arrInputTensors[0]);
				delete[] ImageData;
				if (TF_GetCode(m_Status) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
					m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
			}
		}
	}

	else
	{
		if (bNormalize)
		{
			for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
			{
				int nCurrBatch = nBatch;
				if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
					nCurrBatch = nImage % nBatch;

				float* ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					int nCurrImgIdx = batchIdx * nBatch + dataIdx;
					int nCurrXIdx = nCurrImgIdx % nIterX;
					int nCurrYIdx = nCurrImgIdx / nIterX;

					int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
					int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
					if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
					if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

					for (int y = 0; y < ptCropSize.y; ++y)
					{
						for (int x = 0; x < ptCropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x * nImageChannel + c]) / float(255.);
							}
						}
					}
				}

				m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
				size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

				TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
					m_InputDims[0],
					m_nInputDims[0],
					ImageData,
					InputDataSize,
					Deallocator,
					nullptr);

				arrInputTensors[0] = InputImageTensor;

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				{
					m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
					size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);
					arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
				}

				TF_SessionRun(m_Session, m_RunOptions,
					m_arrInputOps, arrInputTensors, m_nInputOps,
					m_arrOutputOps, arrOutputTensors, m_nOutputOps,
					nullptr, 0, nullptr, m_Status);

				//Input Tensor 메모리 해제
				TF_DeleteTensor(arrInputTensors[0]);
				delete[] ImageData;
				if (TF_GetCode(m_Status) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
					m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
			}
		}
		else
		{
			for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
			{
				int nCurrBatch = nBatch;
				if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
					nCurrBatch = nImage % nBatch;

				float* ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2] * nImageChannel];
				for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
				{
					int nCurrImgIdx = batchIdx * nBatch + dataIdx;
					int nCurrXIdx = nCurrImgIdx % nIterX;
					int nCurrYIdx = nCurrImgIdx / nIterX;

					int nCurrX = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
					int nCurrY = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
					if (nCurrX + m_ptCropSize.x > m_ptImageSize.x) nCurrX = m_ptImageSize.x - m_ptCropSize.x;
					if (nCurrY + m_ptCropSize.y > m_ptImageSize.y) nCurrY = m_ptImageSize.y - m_ptCropSize.y;

					for (int y = 0; y < ptCropSize.y; ++y)
					{
						for (int x = 0; x < ptCropSize.x; ++x)
						{
							for (int c = 0; c < nImageChannel; ++c)
							{
								ImageData[dataIdx * ptCropSize.y * ptCropSize.x * nImageChannel + y * ptCropSize.x * nImageChannel + x * nImageChannel + c] = float(ppImage[ptBuffPos.y + nCurrY + y][ptBuffPos.x + nCurrX + x * nImageChannel + c]);
							}
						}
					}
				}

				m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
				size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

				TF_Tensor* InputImageTensor = TF_NewTensor(TF_FLOAT,
					m_InputDims[0],
					m_nInputDims[0],
					ImageData,
					InputDataSize,
					Deallocator,
					nullptr);

				arrInputTensors[0] = InputImageTensor;

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
				{
					m_OutputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
					size_t OutputDataSize = m_OutputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);
					arrOutputTensors[opsIdx] = TF_AllocateTensor(TF_FLOAT, m_OutputDims[opsIdx], m_nOutputDims[opsIdx], OutputDataSize);
				}

				TF_SessionRun(m_Session, m_RunOptions,
					m_arrInputOps, arrInputTensors, m_nInputOps,
					m_arrOutputOps, arrOutputTensors, m_nOutputOps,
					nullptr, 0, nullptr, m_Status);

				//Input Tensor 메모리 해제
				TF_DeleteTensor(arrInputTensors[0]);
				delete[] ImageData;
				if (TF_GetCode(m_Status) != TF_OK)
				{
					return false;
				}

				for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
					m_vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
			}
		}
	}

	return true;
}

bool TFCore::FreeModel()
{
	for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		delete[] m_InputDims[opsIdx];
	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
		delete[] m_OutputDims[opsIdx];
	delete[] m_arrInputOps;
	delete[] m_arrOutputOps;
	delete[] m_nInputDims;
	delete[] m_nOutputDims;
	delete[] m_InputDataSizePerBatch;
	delete[] m_OutputDataSizePerBatch;
	delete[] m_InputDims;
	delete[] m_OutputDims;

	if (m_RunOptions != nullptr)
		TF_DeleteBuffer(m_RunOptions);
	if (m_MetaGraph != nullptr)
		TF_DeleteBuffer(m_MetaGraph);
	if (m_Session != nullptr)
	{
		TF_CloseSession(m_Session, m_Status);
		TF_DeleteSession(m_Session, m_Status);
	}
	if (m_SessionOptions != nullptr)
		TF_DeleteSessionOptions(m_SessionOptions);
	if (m_Graph != nullptr)
		TF_DeleteGraph(m_Graph);
	if (m_Status != nullptr)
		TF_DeleteStatus(m_Status);

	m_bModelLoaded = false;
	m_bDataLoaded = false;

	return true;
}

bool TFCore::IsModelLoaded()
{
	return m_bModelLoaded;
}

long long** TFCore::GetInputDims()
{
	return m_InputDims;
}

long long** TFCore::GetOutputDims()
{
	return m_OutputDims;
}

bool TFCore::_Run()
{
	return true;
}