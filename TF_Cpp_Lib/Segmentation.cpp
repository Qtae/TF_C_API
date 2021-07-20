#pragma once
#include "Segmentation.h"


Segmentation::Segmentation()
{
}

Segmentation::~Segmentation()
{
}

std::vector<std::vector<float *>> Segmentation::GetOutput()
{
	if (!m_vtOutputRes.empty())
		FreeOutputMap();

	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator °¹¼ö iteration
	{
		std::vector<float*> vtResultOp;
		for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
			int nOutputSize = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1) * (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
			float *pOutput = new float[nBatch * nOutputSize];
			std::memcpy(pOutput, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nOutputSize * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				float* pOutputBatch = new float[nOutputSize];
				std::memcpy(pOutputBatch, pOutput + batchIdx * nOutputSize, nOutputSize * sizeof(float));
				vtResultOp.push_back(pOutputBatch);
			}
			delete[] pOutput;
		}
		m_vtOutputRes.push_back(vtResultOp);
	}
	return m_vtOutputRes;
}

bool Segmentation::FreeOutputMap()
{
	if (!m_vtOutputRes.empty())
	{
		for (int opsIdx = 0; opsIdx < m_vtOutputRes.size(); ++opsIdx)
		{
			for (int nIdx = 0; nIdx < m_vtOutputRes[opsIdx].size(); ++nIdx)
			{
				delete[] m_vtOutputRes[opsIdx][nIdx];
			}
		}
	}
	return true;
}

std::vector<std::vector<int*>> Segmentation::GetWholeClsMask()
{
	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)//output operator °¹¼ö iteration
	{
		std::vector<int*> vtClassMaskOp;
		for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
			int nOutputSize = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1) * (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
			float *pOutput = new float[nBatch * nOutputSize];
			std::memcpy(pOutput, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nOutputSize * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				float* pOutputBatch = new float[nOutputSize];
				std::memcpy(pOutputBatch, pOutput + batchIdx * nOutputSize, nOutputSize * sizeof(float));
			}
			delete[] pOutput;
		}
		m_vtClassMask.push_back(vtClassMaskOp);
	}
	return m_vtClassMask;
}

std::vector<std::vector<int*>> Segmentation::GetBinaryMaskWithClsIndex(int nClsIndex)
{
	for (int opsIdx = 0; opsIdx < m_nOutputOps; ++opsIdx)
	{
		std::vector<int*> vtClassMaskOp;
		for (int i = 0; i < m_vtOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int nBatch = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 0);
			int nOutputSize = (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1) * (int)TF_Dim(m_vtOutputTensors[opsIdx][i], 1);
			float *pOutput = new float[nBatch * nOutputSize];
			std::memcpy(pOutput, TF_TensorData(m_vtOutputTensors[opsIdx][i]), nBatch * nOutputSize * sizeof(float));
			for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
			{
				int* pOutputBatch = new int[nOutputSize];
				for (int pixIdx = 0; pixIdx < nOutputSize; ++pixIdx)
				{
					if ((int)pOutput[batchIdx * nOutputSize + pixIdx] == nClsIndex)
						pOutputBatch[pixIdx] = 1;
					else
						pOutputBatch[pixIdx] = 0;
				}
				vtClassMaskOp.push_back(pOutputBatch);
			}
			delete[] pOutput;
		}
		m_vtClassMask.push_back(vtClassMaskOp);
	}
	return m_vtClassMask;
}

bool Segmentation::GetWholeImageSegmentationResults(unsigned char* pImg, int nClsNo)
{
	//Suppose there is only one output operation in detection tasks.
	std::vector<DetectionResult> vtResult;
	int nBatch = (int)m_OutputDims[0][0];
	int nWidth = (int)m_OutputDims[0][1];
	int nHeight = (int)m_OutputDims[0][2];
	int nChannel = (int)m_OutputDims[0][3];
	if (nClsNo >= nChannel) return false;

	int nIterX = (int)(m_ptImageSize.x / (m_ptCropSize.x - m_ptOverlapSize.x));
	int nIterY = (int)(m_ptImageSize.y / (m_ptCropSize.y - m_ptOverlapSize.y));
	if (m_ptImageSize.x - ((m_ptCropSize.x - m_ptOverlapSize.x) * (nIterX - 1)) > m_ptCropSize.x) ++nIterX;
	if (m_ptImageSize.y - ((m_ptCropSize.y - m_ptOverlapSize.y) * (nIterY - 1)) > m_ptCropSize.y) ++nIterY;
	int nCurrXIdx = 0;
	int nCurrYIdx = 0;
	int nCurrImgIdx = 0;

	for (int i = 0; i < m_vtOutputTensors[0].size(); ++i)//Tensor iteration
	{
		float *output = new float[nBatch * nWidth * nHeight * nChannel];
		std::memcpy(output, TF_TensorData(m_vtOutputTensors[0][i]), nBatch * nWidth * nHeight * nChannel * sizeof(float));

		for (int imgIdx = 0; imgIdx < nBatch; ++imgIdx)//Image in tensor iteration
		{
			nCurrImgIdx = i * nBatch + imgIdx;
			nCurrXIdx = nCurrImgIdx % nIterX;
			nCurrYIdx = nCurrImgIdx / nIterX;
			int nXOffset = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
			int nYOffset = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
			if (nXOffset + m_ptCropSize.x > m_ptImageSize.x) nXOffset = m_ptImageSize.x - m_ptCropSize.x;
			if (nYOffset + m_ptCropSize.y > m_ptImageSize.y) nYOffset = m_ptImageSize.y - m_ptCropSize.y;
			
			for (int y = 0; y < nHeight; ++y)
			{
				for (int x = 0; x < nWidth; ++x)
				{
					pImg[(nYOffset + y) * m_ptImageSize.x + nXOffset + x] = output[imgIdx * nHeight * nWidth * nChannel + y * nWidth * nChannel + x * nChannel + nClsNo];
				}
			}
		}
		delete[] output;
	}


	for (int opsIdx = 0; opsIdx < m_nInputOps; ++opsIdx)
	{
		for (int tensorIdx = 0; tensorIdx < m_vtOutputTensors[opsIdx].size(); ++tensorIdx)
		{
			TF_DeleteTensor(m_vtOutputTensors[opsIdx][tensorIdx]);
		}
		m_vtOutputTensors[opsIdx].clear();
	}
	m_vtOutputTensors.clear();

	return true;
}