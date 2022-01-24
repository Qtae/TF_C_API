#pragma once
#include <atltypes.h>
#include "TFTool.h"
#include "Classification.h"
#include "Segmentation.h"
#include "Detection.h"


namespace TFTool
{
	AI::AI()
	{
		mTaskType = -1;
		mClassification = new Classification();
		mSegmentation = new Segmentation();
		mDetection = new Detection();
	}

	AI::~AI()
	{

	}

	bool AI::LoadModel(const char* modelPath, std::vector<const char*> &inputOpNames, std::vector<const char*>& outputOpNames, int taskType)
	{
		mTaskType = taskType;
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->LoadModel(modelPath, inputOpNames, outputOpNames);
			break;
		case 1://segmentation
			res = mSegmentation->LoadModel(modelPath, inputOpNames, outputOpNames);
			break;
		case 2://detection
			res = mDetection->LoadModel(modelPath, inputOpNames, outputOpNames);
			break;
		}
		return res;
	}

	bool AI::Run(float** inputImgArr, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(float*** inputImgArr, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char** inputImgArr, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char*** inputImgArr, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY, bool bNormalize, bool bConvertGrayToColor)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor);
			break;
		case 2://detection
			res = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor);
			break;
		}
		return res;
	}

	bool AI::Run(float*** inputImgArr, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(float** inputImgArr, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char*** inputImgArr, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char** inputImgArr, int batch, bool bNormalize)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			res = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return res;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY, int batch, bool bNormalize, bool bConvertGrayToColor)
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGrayToColor);
			break;
		case 1://segmentation
			res = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGrayToColor);
			break;
		case 2://detection
			res = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGrayToColor);
			break;
		}
		return res;
	}

	bool AI::FreeModel()
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->FreeModel();
			break;
		case 1://segmentation
			res = mSegmentation->FreeModel();
			break;
		case 2://detection
			res = mDetection->FreeModel();
			break;
		}
		return res;
	}

	bool AI::GetClassificationResults(float*** classificationResultArray)
	{
		if (mTaskType != 0)
			return false;
		else
		{
			mClassification->GetOutput(classificationResultArray);
		}
	}

	std::vector<std::vector<int>> AI::GetClassificationResults(float softmxThresh)
	{
		std::vector<std::vector<int>> result;
		if (mTaskType != 0)
			return result;
		else
		{
			result = mClassification->GetPredCls(softmxThresh);
			return result;
		}
	}

	std::vector<std::vector<float*>> AI::GetSegmentationResults()
	{
		std::vector<std::vector<float*>> result;
		if (mTaskType != 1)
			return result;
		else
		{
			result = mSegmentation->GetOutput();
			return result;
		}
	}

	std::vector<std::vector<DetectionResult>> AI::GetDetectionResults(float iouThresh, float scoreThresh)
	{
		std::vector<std::vector<DetectionResult>> result;
		if (mTaskType != 2)
			return result;
		else
		{
			result = mDetection->GetDetectionResults(iouThresh, scoreThresh);
			return result;
		}
	}

	bool AI::GetDetectionResultsByArray(DetectionResult** detectionResultArr, int* boxNumArr, float iouThresh, float scoreThresh)
	{
		if (mTaskType != 2)
			return false;
		else
		{
			bool res = mDetection->GetDetectionResults(detectionResultArr, boxNumArr, iouThresh, scoreThresh);
			return res;
		}
	}

	bool AI::GetWholeImageDetectionResults(DetectionResult* detResArr, int& boxNum, float iouThresh, float scoreThresh)
	{
		if (mTaskType != 2)
			return false;
		else
		{
			bool res = mDetection->GetWholeImageDetectionResultsSingleOutput(detResArr, boxNum, iouThresh, scoreThresh);
			return res;
		}
	}

	bool AI::GetWholeImageSegmentationResults(unsigned char* outputImg, int clsNo)
	{
		if (mTaskType != 1)
			return false;
		else
		{
			bool res = mSegmentation->GetWholeImageSegmentationResults(outputImg, clsNo);
			return res;
		}
	}

	bool AI::IsModelLoaded()
	{
		bool res = false;
		switch (mTaskType)
		{
		case 0://classification
			res = mClassification->IsModelLoaded();
			break;
		case 1://segmentation
			res = mSegmentation->IsModelLoaded();
			break;
		case 2://detection
			res = mDetection->IsModelLoaded();
			break;
		}
		return res;
	}

	long long** AI::GetInputDims()
	{
		long long** dims = NULL;
		switch (mTaskType)
		{
		case 0://classification
			dims = mClassification->GetInputDims();
			break;
		case 1://segmentation
			dims = mSegmentation->GetInputDims();
			break;
		case 2://detection
			dims = mDetection->GetInputDims();
			break;
		}
		return dims;
	}

	long long** AI::GetOutputDims()
	{
		long long** dims = NULL;
		switch (mTaskType)
		{
		case 0://classification
			dims = mClassification->GetOutputDims();
			break;
		case 1://segmentation
			dims = mSegmentation->GetOutputDims();
			break;
		case 2://detection
			dims = mDetection->GetOutputDims();
			break;
		}
		return dims;
	}
}