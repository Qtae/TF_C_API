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
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->LoadModel(modelPath, inputOpNames, outputOpNames);
			break;
		case 1://segmentation
			bRes = mSegmentation->LoadModel(modelPath, inputOpNames, outputOpNames);
			break;
		case 2://detection
			bRes = mDetection->LoadModel(modelPath, inputOpNames, outputOpNames);
			break;
		}
		return bRes;
	}

	bool AI::Run(float** inputImgArr, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(float*** inputImgArr, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** inputImgArr, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char*** inputImgArr, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY, bool bNormalize, bool bConvertGrayToColor)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), bNormalize, bConvertGrayToColor);
			break;
		}
		return bRes;
	}

	bool AI::Run(float*** inputImgArr, int batch, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(float** inputImgArr, int batch, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char*** inputImgArr, int batch, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** inputImgArr, int batch, bool bNormalize)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImgArr, batch, bNormalize);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImgArr, batch, bNormalize);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImgArr, batch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** inputImg, int imgSizeX, int imgSizeY, int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY, int batch, bool bNormalize, bool bConvertGrayToColor)
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGrayToColor);
			break;
		case 1://segmentation
			bRes = mSegmentation->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGrayToColor);
			break;
		case 2://detection
			bRes = mDetection->Run(inputImg, CPoint(imgSizeX, imgSizeY), CPoint(cropSizeX, cropSizeY), CPoint(overlapSizeX, overlapSizeY), CPoint(buffPosX, buffPosY), batch, bNormalize, bConvertGrayToColor);
			break;
		}
		return bRes;
	}

	bool AI::FreeModel()
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->FreeModel();
			break;
		case 1://segmentation
			bRes = mSegmentation->FreeModel();
			break;
		case 2://detection
			bRes = mDetection->FreeModel();
			break;
		}
		return bRes;
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

	std::vector<std::vector<std::vector<float>>> AI::GetClassificationSoftMXResults()
	{
		std::vector<std::vector<std::vector<float>>> result;
		if (mTaskType != 0)
			return result;
		else
		{
			result = mClassification->GetSoftMXResult();
			return result;
		}
	}

	bool AI::GetSegmentationResults(float*** InputArr)
	{
		bool result= FALSE;
		if (mTaskType != 1)
			return result;
		else
		{
			result = mSegmentation->GetOutput(InputArr);
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

	bool AI::GetWholeImageDetectionResults(DetectionResult* detResArr, int& boxNum, float iouThresh, float scoreThresh)
	{
		if (mTaskType != 2)
			return false;
		else
		{
			bool bRes = mDetection->GetWholeImageDetectionResultsSingleOutput(detResArr, boxNum, iouThresh, scoreThresh);
			return bRes;
		}
	}

	bool AI::GetWholeImageSegmentationResults(unsigned char* outputImg, int clsNo)
	{
		if (mTaskType != 1)
			return false;
		else
		{
			bool bRes = mSegmentation->GetWholeImageSegmentationResults(outputImg, clsNo);
			return bRes;
		}
	}

	bool AI::IsModelLoaded()
	{
		bool bRes = false;
		switch (mTaskType)
		{
		case 0://classification
			bRes = mClassification->IsModelLoaded();
			break;
		case 1://segmentation
			bRes = mSegmentation->IsModelLoaded();
			break;
		case 2://detection
			bRes = mDetection->IsModelLoaded();
			break;
		}
		return bRes;
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