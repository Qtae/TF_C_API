#pragma once
#include <iostream>
#include <atltypes.h>
#include <Windows.h>
#include <array>
#include <vector>
#include <string>
#include <time.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "pch.h"


class TFCore
{
public:
	TFCore();
	~TFCore();

	bool LoadModel(const char*, std::vector<const char*>&, std::vector<const char*>&);

	bool Run(float**, bool bNormalize = false);
	bool Run(float***, bool bNormalize = false);
	bool Run(unsigned char**, bool bNormalize = false);
	bool Run(unsigned char***, bool bNormalize = false);
	bool Run(std::vector<std::vector<cv::Mat>>, bool bNormalize = false);
	bool Run(std::vector<cv::Mat>, bool bNormalize = false);

	bool Run(float***, int, bool bNormalize = false);
	bool Run(float**, int, bool bNormalize = false);
	bool Run(unsigned char***, int, bool bNormalize = false);
	bool Run(unsigned char**, int, bool bNormalize = false);
	bool Run(std::vector<std::vector<cv::Mat>>, int, bool bNormalize = false);
	bool Run(std::vector<cv::Mat>, int, bool bNormalize = false);

	//VisionWorks image input format, has only one input operator
	bool Run(unsigned char**, CPoint, CPoint,int, bool bNormalize = false);

	bool FreeModel();

private:
	bool _Run();

protected:
	const char* m_ModelPath;
	std::vector<const char*> m_vtClassNames;

	bool m_bModelLoaded;
	bool m_bDataLoaded;

	TF_Session* m_Session;
	TF_Buffer* m_RunOptions;
	TF_SessionOptions* m_SessionOptions;
	TF_Graph* m_Graph;
	TF_Status* m_Status;
	TF_Buffer* m_MetaGraph;

	int m_nInputOps;
	int m_nOutputOps;
	TF_Output* m_arrInputOps;
	TF_Output* m_arrOutputOps;

	int* m_nInputDims;
	int* m_nOutputDims;
	long long** m_InputDims;
	long long** m_OutputDims;
	std::size_t* m_InputDataSizePerBatch;
	std::size_t* m_OutputDataSizePerBatch;

	std::vector<std::vector<TF_Tensor*>> m_vtOutputTensors;
};