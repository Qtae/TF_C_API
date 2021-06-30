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
		m_nTaskType = -1;
		pClassification = new Classification();
		pSegmentation = new Segmentation();
		pDetection = new Detection();
	}

	AI::~AI()
	{

	}

	bool AI::LoadModel(const char* ModelPath, std::vector<const char*> &vtInputOpNames, std::vector<const char*>& vtOutputOpNames, int nTaskType)
	{
		m_nTaskType = nTaskType;
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->LoadModel(ModelPath, vtInputOpNames, vtOutputOpNames);
			break;
		case 1://segmentation
			bRes = pSegmentation->LoadModel(ModelPath, vtInputOpNames, vtOutputOpNames);
			break;
		case 2://detection
			bRes = pDetection->LoadModel(ModelPath, vtInputOpNames, vtOutputOpNames);
			break;
		}
		return bRes;
	}

	bool AI::Run(float** pImageSet, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(float*** pImageSet, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** pImageSet, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char*** pImageSet, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(float*** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(float** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char*** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::Run(unsigned char** ppImage, int nCropSizeX, int nCropSizeY, int nOverlapSizeX, int nOverlapSizeY, int nBatch, bool bNormalize)
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(ppImage, CPoint(nCropSizeX, nCropSizeY), CPoint(nOverlapSizeX, nOverlapSizeY), nBatch, bNormalize);
			break;
		case 1://segmentation
			bRes = pSegmentation->Run(ppImage, CPoint(nCropSizeX, nCropSizeY), CPoint(nOverlapSizeX, nOverlapSizeY), nBatch, bNormalize);
			break;
		case 2://detection
			bRes = pDetection->Run(ppImage, CPoint(nCropSizeX, nCropSizeY), CPoint(nOverlapSizeX, nOverlapSizeY), nBatch, bNormalize);
			break;
		}
		return bRes;
	}

	bool AI::FreeModel()
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->FreeModel();
			break;
		case 1://segmentation
			bRes = pSegmentation->FreeModel();
			break;
		case 2://detection
			bRes = pDetection->FreeModel();
			break;
		}
		return bRes;
	}

	std::vector<std::vector<std::vector<float>>> AI::GetClassificationResults()
	{
		std::vector<std::vector<std::vector<float>>> vtResult;
		if (m_nTaskType != 0) 
			return vtResult;
		else
		{
			vtResult = pClassification->GetOutput();
			return vtResult;
		}
	}

	std::vector<std::vector<int>> AI::GetClassificationResults(float fSoftmxThresh)
	{
		std::vector<std::vector<int>> vtResult;
		if (m_nTaskType != 0)
			return vtResult;
		else
		{
			vtResult = pClassification->GetPredCls(fSoftmxThresh);
			return vtResult;
		}
	}

	std::vector<std::vector<float*>> AI::GetSegmentationResults()
	{
		std::vector<std::vector<float*>> vtResult;
		if (m_nTaskType != 1)
			return vtResult;
		else
		{
			vtResult = pSegmentation->GetOutput();
			return vtResult;
		}
	}

	std::vector<std::vector<DetectionResult>> AI::GetDetectionResults(float fIOUThres, float fScoreThres)
	{
		std::vector<std::vector<DetectionResult>> vtResult;
		if (m_nTaskType != 2)
			return vtResult;
		else
		{
			vtResult = pDetection->GetDetectionResults(fIOUThres, fScoreThres);
			return vtResult;
		}
	}

	bool AI::IsModelLoaded()
	{
		bool bRes = false;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->IsModelLoaded();
			break;
		case 1://segmentation
			bRes = pSegmentation->IsModelLoaded();
			break;
		case 2://detection
			bRes = pDetection->IsModelLoaded();
			break;
		}
		return bRes;
	}

	long long** AI::GetInputDims()
	{
		long long** pDims = NULL;
		switch (m_nTaskType)
		{
		case 0://classification
			pDims = pClassification->GetInputDims();
			break;
		case 1://segmentation
			pDims = pSegmentation->GetInputDims();
			break;
		case 2://detection
			pDims = pDetection->GetInputDims();
			break;
		}
		return pDims;
	}

	long long** AI::GetOutputDims()
	{
		long long** pDims = NULL;
		switch (m_nTaskType)
		{
		case 0://classification
			pDims = pClassification->GetOutputDims();
			break;
		case 1://segmentation
			pDims = pSegmentation->GetOutputDims();
			break;
		case 2://detection
			pDims = pDetection->GetOutputDims();
			break;
		}
		return pDims;
	}
}