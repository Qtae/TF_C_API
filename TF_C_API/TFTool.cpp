#pragma once
#include "TFTool.h"
#include "Classification.h"
#include "Segmentation.h"
#include "Detection.h"


namespace TFTool
{
	int m_nTaskType = -1;
	Classification* pClassification = new Classification();
	Segmentation* pSegmentation = new Segmentation();
	Detection* pDetection = new Detection();

	bool LoadModel(const char* ModelPath, std::vector<const char*> &vtInputOpNames, std::vector<const char*>& vtOutputOpNames, int nTaskType)
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

	bool Run(float** pImageSet, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return true;
	}

	bool Run(float*** pImageSet, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return true;
	}

	bool Run(unsigned char** pImageSet, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return true;
	}

	bool Run(unsigned char*** pImageSet, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, bNormalize);
			break;
		}
		return true;
	}

	bool Run(float*** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return true;
	}

	bool Run(float** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return true;
	}

	bool Run(unsigned char*** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return true;
	}

	bool Run(unsigned char** pImageSet, int nBatch, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(pImageSet, nBatch, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(pImageSet, nBatch, bNormalize);
			break;
		case 2://detection
			pDetection->Run(pImageSet, nBatch, bNormalize);
			break;
		}
		return true;
	}

	bool Run(unsigned char** ppImage, CPoint ptCropSize, CPoint ptOverlapSize, int nBatch, bool bNormalize)
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->Run(ppImage, ptCropSize, ptOverlapSize, nBatch, bNormalize);
			break;
		case 1://segmentation
			pSegmentation->Run(ppImage, ptCropSize, ptOverlapSize, nBatch, bNormalize);
			break;
		case 2://detection
			pDetection->Run(ppImage, ptCropSize, ptOverlapSize, nBatch, bNormalize);
			break;
		}
		return true;
	}

	bool FreeModel()
	{
		bool bRes;
		switch (m_nTaskType)
		{
		case 0://classification
			bRes = pClassification->FreeModel();
			break;
		case 1://segmentation
			pSegmentation->FreeModel();
			break;
		case 2://detection
			pDetection->FreeModel();
			break;
		}
		return true;
	}

	std::vector<std::vector<std::vector<float>>> GetClassificationResults()
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

	std::vector<std::vector<int>> GetClassificationResults(float fSoftmxThresh)
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

	std::vector<std::vector<float*>> GetSegmentationResults()
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

	std::vector<std::vector<DetectionResult>> GetDetectionResults(float fIOUThres, float fScoreThres)
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

}