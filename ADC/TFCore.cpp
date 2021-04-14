#include "TFCore.h"


TFCore::TFCore()
{
	m_bModelLoaded = false;
	m_bDataLoaded = false;
}

TFCore::~TFCore()
{
	FreeModel();
}

bool TFCore::LoadModel(const char *ModelPath, std::vector<const char *> &vtInputOpNames, std::vector<const char *> &vtOutputOpNames)
{
	m_ModelPath = ModelPath;
	m_RunOptions = TF_NewBufferFromString("", 0);
	m_SessionOptions = TF_NewSessionOptions();
	m_Graph = TF_NewGraph();
	m_Status = TF_NewStatus();
	m_MetaGraph = TF_NewBuffer();
	m_Session = TF_NewSession(m_Graph, m_SessionOptions, m_Status);
	const char *tag = "serve";
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
		TF_Operation *InputOp = TF_GraphOperationByName(m_Graph, vtInputOpNames[i]);
		if (InputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		m_arrInputOps[i] = TF_Output{ InputOp, 0 };
		m_nInputDims[i] = TF_GraphGetTensorNumDims(m_Graph, m_arrInputOps[0], m_Status);
		int64_t *InputShape = new int64_t[m_nInputDims[i]];
		TF_GraphGetTensorShape(m_Graph, m_arrInputOps[0], InputShape, m_nInputDims[i], m_Status);
		long long * InputDims = new long long[m_nInputDims[i]];
		m_InputDims[i] = InputDims;
		m_InputDims[i][0] = static_cast<long long>(1);
		for (int j = 1; j < m_nInputDims[i]; ++j) m_InputDims[i][j] = static_cast<long long>(InputShape[j]);
		m_InputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(m_arrInputOps[i]));
		for (int j = 1; j < m_nInputDims[i]; ++j) m_InputDataSizePerBatch[i] = m_InputDataSizePerBatch[i] * static_cast<int>(InputShape[j]);
	}

	for (int i = 0; i < m_nOutputOps; ++i)
	{
		TF_Operation *OutputOp = TF_GraphOperationByName(m_Graph, vtOutputOpNames[i]);
		if (OutputOp == nullptr)
		{
			std::cout << "Failed to find graph operation" << std::endl;
			return false;
		}
		m_arrOutputOps[i] = TF_Output{ OutputOp, 0 };
		m_nOutputDims[i] = TF_GraphGetTensorNumDims(m_Graph, m_arrOutputOps[0], m_Status);
		int64_t *OutputShape = new int64_t[m_nOutputDims[i]];
		TF_GraphGetTensorShape(m_Graph, m_arrOutputOps[0], OutputShape, m_nOutputDims[i], m_Status);
		long long * OutputDims = new long long[m_nOutputDims[i]];
		m_OutputDims[i] = OutputDims;
		m_OutputDims[i][0] = static_cast<long long>(1);
		for (int j = 1; j < m_nOutputDims[i]; ++j) m_OutputDims[i][j] = static_cast<long long>(OutputShape[j]);
		m_OutputDataSizePerBatch[i] = TF_DataTypeSize(TF_OperationOutputType(m_arrOutputOps[0]));
		for (int j = 1; j < m_nOutputDims[i]; ++j) m_OutputDataSizePerBatch[i] = m_OutputDataSizePerBatch[i] * static_cast<int>(OutputShape[j]);
	}

	m_bModelLoaded = true;
	return true;
}

bool TFCore::Run(float ***ppImageSet, int nImage, int nBatch)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	TF_Tensor **arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor **arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if(!(vtOutputTensors.empty())) vtOutputTensors.clear();
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor *> vt;
		vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void *, std::size_t, void *) {};

	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
	{
		int nCurrBatch = nBatch;
		if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
			nCurrBatch = nImage % nBatch;
		for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
		{
			float *ImageData = new float[nCurrBatch * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]];//opsIdx 바뀌어도 주소가 안바뀌는지
			for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
			{
				for (int pixIdx = 0; pixIdx < m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2]; ++pixIdx)
				{
					ImageData[dataIdx * m_InputDims[opsIdx][1] * m_InputDims[opsIdx][2] + pixIdx] = ppImageSet[batchIdx * nBatch + dataIdx][opsIdx][pixIdx] / 255.;
				}
			}
			m_InputDims[opsIdx][0] = static_cast<long long>(nCurrBatch);
			size_t InputDataSize = m_InputDataSizePerBatch[opsIdx] * static_cast<size_t>(nCurrBatch);

			TF_Tensor *InputImageTensor = TF_NewTensor(TF_FLOAT,
													   m_InputDims[opsIdx],
													   m_nInputDims[opsIdx],
													   ImageData,
													   InputDataSize,
													   Deallocator,
													   nullptr);
			arrInputTensors[opsIdx] = InputImageTensor;
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

		if (TF_GetCode(m_Status) != TF_OK)
		{
			//Input/Output Tensor 메모리 해제 추가
			return false;
		}

		//Input Tensor 및 Tensor 안의 Data 메모리 해제 추가

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
	}
	return true;
}

bool TFCore::Run(float **ppImageSet, int nImage, int nBatch)
{
	if (!m_bModelLoaded)
	{
		std::cout << "Model Not Loaded!" << std::endl;
		return false;
	}

	TF_Tensor **arrInputTensors = new TF_Tensor*[m_nInputOps];
	TF_Tensor **arrOutputTensors = new TF_Tensor*[m_nOutputOps];

	if (!(vtOutputTensors.empty())) vtOutputTensors.clear();
	for (int idxOutputOps = 0; idxOutputOps < m_nOutputOps; ++idxOutputOps)
	{
		std::vector<TF_Tensor *> vt;
		vtOutputTensors.push_back(vt);
	}

	auto const Deallocator = [](void *, std::size_t, void *) {};

	int nBatchIter = nImage / nBatch + (int)(bool)(nImage % nBatch);

	for (int batchIdx = 0; batchIdx < nBatchIter; ++batchIdx)
	{
		int nCurrBatch = nBatch;
		if ((batchIdx == nBatchIter - 1) && (nImage % nBatch != 0))
			nCurrBatch = nImage % nBatch;

		float *ImageData = new float[nCurrBatch * m_InputDims[0][1] * m_InputDims[0][2]];//opsIdx 바뀌어도 주소가 안바뀌는지
		for (int dataIdx = 0; dataIdx < nCurrBatch; ++dataIdx)
		{
			for (int pixIdx = 0; pixIdx < m_InputDims[0][1] * m_InputDims[0][2]; ++pixIdx)
			{
				ImageData[dataIdx * m_InputDims[0][1] * m_InputDims[0][2] + pixIdx] = ppImageSet[batchIdx * nBatch + dataIdx][pixIdx] / 255.;
			}
		}
		m_InputDims[0][0] = static_cast<long long>(nCurrBatch);
		size_t InputDataSize = m_InputDataSizePerBatch[0] * static_cast<size_t>(nCurrBatch);

		TF_Tensor *InputImageTensor = TF_NewTensor(TF_FLOAT,
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

		if (TF_GetCode(m_Status) != TF_OK)
		{
			//Input/Output Tensor 메모리 해제 추가
			return false;
		}

		//Input Tensor 및 Tensor 안의 Data 메모리 해제 추가

		for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
			vtOutputTensors[opsIdx].push_back(arrOutputTensors[opsIdx]);
	}
	return true;
}

bool TFCore::FreeModel()
{
	if (m_RunOptions != nullptr) TF_DeleteBuffer(m_RunOptions);
	if (m_SessionOptions != nullptr) TF_DeleteSessionOptions(m_SessionOptions);
	if (m_Session != nullptr) TF_DeleteSession(m_Session, m_Status);
	if (m_Graph != nullptr) TF_DeleteGraph(m_Graph);
	if (m_Status != nullptr) TF_DeleteStatus(m_Status);

	m_bModelLoaded = false;
	m_bDataLoaded = false;

	//다른 동적배열도 메모리 해제 필요

	return true;
}