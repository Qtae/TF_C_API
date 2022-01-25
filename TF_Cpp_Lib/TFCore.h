#pragma once
#include <iostream>
#include <atltypes.h>
#include <Windows.h>
#include <array>
#include <vector>
#include <string>
#include <time.h>
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "TFResultStructure.h"


class TFCore
{
protected:
	const char* mModelPath;

	bool mIsModelLoaded;
	bool mIsDataLoaded;

	TF_Session* mSession;
	TF_Buffer* mRunOptions;
	TF_SessionOptions* mSessionOptions;
	TF_Graph* mGraph;
	TF_Status* mStatus;
	TF_Buffer* mMetaGraph;

	int mInputOpNum;
	int mOutputOpNum;
	TF_Output* mInputOpsArr;
	TF_Output* mOutputOpsArr;

	int* mInputDims; //ex) mInputDims[0] = 4
	int* mOutputDims; //ex) mOutputDims[0] = 4 mOutputDims[1] = 2
	long long** mInputDimsArr; //ex) mInputDimsArr[0][1] =640  mInputDimsArr[1] = [-1, 640, 640, 1]
	long long** mOutputDimsArr;
	std::size_t* mInputDataSizePerBatch;
	std::size_t* mOutputDataSizePerBatch;

	std::vector<std::vector<TF_Tensor*>> mOutputTensors;

	CPoint mImageSize;
	CPoint mCropSize;
	CPoint mOverlapSize;

public:
	TFCore();
	~TFCore();

	bool LoadModel(const char*, std::vector<const char*>&, std::vector<const char*>&);

	bool Run(float**, bool bNormalize = false);
	bool Run(float***, bool bNormalize = false);
	bool Run(unsigned char**, bool bNormalize = false);
	bool Run(unsigned char***, bool bNormalize = false);
	bool Run(std::vector<std::vector<cv::Mat>>, bool bNormalize = false);
	bool Run(std::vector<cv::Mat>, bool bNormalize = false);//

	//VisionWorks image input format, has only one input operator
	bool Run(unsigned char**, CPoint, CPoint, CPoint, CPoint, bool bNormalize = false, bool bConvertGrayToColor = false);

	bool Run(float***, int, bool bNormalize = false);
	bool Run(float**, int, bool bNormalize = false);
	bool Run(unsigned char***, int, bool bNormalize = false);
	bool Run(unsigned char**, int, bool bNormalize = false);
	bool Run(std::vector<std::vector<cv::Mat>>, int, bool bNormalize = false);
	bool Run(std::vector<cv::Mat>, int, bool bNormalize = false);

	//VisionWorks image input format, has only one input operator
	bool Run(unsigned char**, CPoint, CPoint, CPoint, CPoint, int, bool bNormalize = false, bool bConvertGrayToColor = false);
	//bool Run(unsigned char**, CPoint, CPoint, int, bool bNormalize = false);

	bool FreeModel();

	bool IsModelLoaded();

	long long** GetInputDims();
	long long** GetOutputDims();

private:
	bool _Run();
};