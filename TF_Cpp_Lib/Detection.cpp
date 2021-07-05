#pragma once
#include "Detection.h"

Detection::Detection()
{
}

Detection::~Detection()
{
}

bool CompareScore(DetectionResult x, DetectionResult y)
{
	if (x.Score * x.Objectness > y.Score * y.Objectness) return true;
	else return false;
}

float Detection::CalculateIOU(DetectionResult Box1, DetectionResult Box2)
{
	int MaxX = std::max(Box1.x - (Box1.w / 2), Box2.x - (Box2.w / 2));
	int MaxY = std::max(Box1.y - (Box1.h / 2), Box2.y - (Box2.h / 2));
	int MinX = std::min(Box1.x + (Box1.w / 2), Box2.x + (Box2.w / 2));
	int MinY = std::min(Box1.y + (Box1.h / 2), Box2.y + (Box2.h / 2));
	int OvelapWidth = ((MinX - MaxX + 1) > 0) ? (MinX - MaxX + 1) : 0;
	int OverlapHeight = ((MinY - MaxY + 1) > 0) ? (MinY - MaxY + 1) : 0;
	int OverlapArea = OvelapWidth * OverlapHeight;
	int Box1Area = Box1.h * Box1.w;
	int Box2Area = Box2.h * Box2.w;
	return float(OverlapArea) / float(Box1Area + Box2Area - OverlapArea);
}

void Detection::DoNMS(std::vector<DetectionResult>& vtDetRes, float fIOUThres, float fScoreThres, int nClass)
{
	if (vtDetRes.empty()) return;
	std::vector<std::vector<DetectionResult>> vtDetResOfEachClass;
	for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
	{
		std::vector<DetectionResult> tmp;
		for (std::vector<DetectionResult>::iterator it = vtDetRes.begin(); it != vtDetRes.end();)
		{
			if ((*it).BestClass == clsIdx)
			{
				tmp.push_back(*it);
				it = vtDetRes.erase(it);
			}
			else it++;
		}
		ApplyScoreThreshold(tmp, fScoreThres);
		sort(tmp.begin(), tmp.end(), CompareScore);//sort the candidate boxes by confidence
		vtDetResOfEachClass.push_back(tmp);
	}

	for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
	{
		for (int i = 0; i < vtDetResOfEachClass[clsIdx].size(); i++)
		{
			if (vtDetResOfEachClass[clsIdx][i].Score > 0)
			{
				for (int j = i + 1; j < vtDetResOfEachClass[clsIdx].size(); j++)
				{
					if (vtDetResOfEachClass[clsIdx][j].Score > 0)
					{
						float iou = CalculateIOU(vtDetResOfEachClass[clsIdx][i], vtDetResOfEachClass[clsIdx][j]);//calculate the orthogonal ratio
						if (iou > fIOUThres) vtDetResOfEachClass[clsIdx][j].Score = 0;
					}
				}
			}
		}
		for (std::vector<DetectionResult>::iterator it = vtDetResOfEachClass[clsIdx].begin(); it != vtDetResOfEachClass[clsIdx].end(); ++it)
		{
			if ((*it).Score != 0) vtDetRes.push_back(*it);
		}
	}
	return;
}

void Detection::ApplyScoreThreshold(std::vector<DetectionResult>& vtDetRes, float fScoreThres)
{
	for (std::vector<DetectionResult>::iterator it = vtDetRes.begin(); it != vtDetRes.end();)
	{
		if ((*it).Objectness * (*it).Score < fScoreThres) it = vtDetRes.erase(it);
		else it++;
	}
	return;
}

std::vector<std::vector<DetectionResult>> Detection::GetDetectionResults(float fIOUThres, float fScoreThres)
{
	//Suppose there is only one output operation in detection tasks.
	std::vector<std::vector<DetectionResult>> vtResult;
	int nBatch = (int)m_OutputDims[0][0];
	int nGridX = (int)m_OutputDims[0][1];
	int nGridY = (int)m_OutputDims[0][2];
	int nAnchor = (int)m_OutputDims[0][3];
	int nClass = (int)m_OutputDims[0][4] - 5;

	for (int i = 0; i < m_vtOutputTensors[0].size(); ++i)//Tensor iteration
	{
		float *output = new float[nBatch * nGridX * nGridY * nAnchor * (nClass + 5)];
		std::memcpy(output, TF_TensorData(m_vtOutputTensors[0][i]), nBatch * nGridX * nGridY * nAnchor * (nClass + 5) * sizeof(float));

		for (int imgIdx = 0; imgIdx < nBatch; ++imgIdx)
		{
			std::vector<DetectionResult> vtImgResult;
			for (int grdXIdx = 0; grdXIdx < nGridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < nGridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < nAnchor; ++ancIdx)
					{
						DetectionResult DetRes;
						DetRes.x = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   0];
						DetRes.y = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   1];
						DetRes.w = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   2];
						DetRes.h = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   3];
						DetRes.Objectness = output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
												   grdXIdx * nGridY * nAnchor * (nClass + 5) +
												   grdYIdx * nAnchor * (nClass + 5) +
												   ancIdx * (nClass + 5) +
												   4];
						int nBestClass = -1;
						float fScore = 0.;
						for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
						{
							float fCurrScore = output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5)
								+ grdXIdx * nGridY * nAnchor * (nClass + 5)
								+ grdYIdx * nAnchor * (nClass + 5)
								+ ancIdx * (nClass + 5)
								+ 5 + clsIdx];
							if (fCurrScore >= fScore)
							{
								nBestClass = clsIdx;
								fScore = fCurrScore;
							}
						}
						DetRes.BestClass = nBestClass;
						DetRes.Score = fScore;
						vtImgResult.push_back(DetRes);
					}
				}
			}
			DoNMS(vtImgResult, fIOUThres, fScoreThres, nClass);

			vtResult.push_back(vtImgResult);
		}
	}

	return vtResult;
}

std::vector<DetectionResult> Detection::GetWholeImageDetectionResults(float fIOUThres, float fScoreThres)
{
	//Suppose there is only one output operation in detection tasks.
	std::vector<DetectionResult> vtResult;
	int nBatch = (int)m_OutputDims[0][0];
	int nGridX = (int)m_OutputDims[0][1];
	int nGridY = (int)m_OutputDims[0][2];
	int nAnchor = (int)m_OutputDims[0][3];
	int nClass = (int)m_OutputDims[0][4] - 5;

	int nIterX = (int)(m_ptImageSize.x / (m_ptCropSize.x - m_ptOverlapSize.x));
	int nIterY = (int)(m_ptImageSize.y / (m_ptCropSize.y - m_ptOverlapSize.y));
	if (m_ptImageSize.x - ((m_ptCropSize.x - m_ptOverlapSize.x) * (nIterX - 1)) > m_ptCropSize.x) ++nIterX;
	if (m_ptImageSize.y - ((m_ptCropSize.y - m_ptOverlapSize.y) * (nIterY - 1)) > m_ptCropSize.y) ++nIterY;
	int nCurrXIdx = 0;
	int nCurrYIdx = 0;
	int nCurrImgIdx = 0;

	for (int i = 0; i < m_vtOutputTensors[0].size(); ++i)//Tensor iteration
	{
		float *output = new float[nBatch * nGridX * nGridY * nAnchor * (nClass + 5)];
		std::memcpy(output, TF_TensorData(m_vtOutputTensors[0][i]), nBatch * nGridX * nGridY * nAnchor * (nClass + 5) * sizeof(float));

		for (int imgIdx = 0; imgIdx < nBatch; ++imgIdx)//Image in tensor iteration
		{
			nCurrImgIdx = i * nBatch + imgIdx;
			nCurrXIdx = nCurrImgIdx % nIterX;
			nCurrYIdx = nCurrImgIdx / nIterX;
			for (int grdXIdx = 0; grdXIdx < nGridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < nGridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < nAnchor; ++ancIdx)
					{
						int nXOffset = (m_ptCropSize.x - m_ptOverlapSize.x) * nCurrXIdx;
						int nYOffset = (m_ptCropSize.y - m_ptOverlapSize.y) * nCurrYIdx;
						if (nXOffset > m_ptImageSize.x) nXOffset = m_ptImageSize.x - m_ptCropSize.x;
						if (nYOffset > m_ptImageSize.y) nYOffset = m_ptImageSize.y - m_ptCropSize.y;

						DetectionResult DetRes;
						DetRes.x = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   0] + 
								   nXOffset;
						DetRes.y = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   1] +
								   nYOffset;
						DetRes.w = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   2];
						DetRes.h = (int)output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
											   grdXIdx * nGridY * nAnchor * (nClass + 5) + 
											   grdYIdx * nAnchor * (nClass + 5) + 
											   ancIdx * (nClass + 5) + 
											   3];
						DetRes.Objectness = output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5) + 
												   grdXIdx * nGridY * nAnchor * (nClass + 5) +
												   grdYIdx * nAnchor * (nClass + 5) +
												   ancIdx * (nClass + 5) +
												   4];
						int nBestClass = -1;
						float fScore = 0.;
						for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
						{
							float fCurrScore = output[imgIdx * nGridX * nGridY * nAnchor * (nClass + 5)
								+ grdXIdx * nGridY * nAnchor * (nClass + 5)
								+ grdYIdx * nAnchor * (nClass + 5)
								+ ancIdx * (nClass + 5)
								+ 5 + clsIdx];
							if (fCurrScore >= fScore)
							{
								nBestClass = clsIdx;
								fScore = fCurrScore;
							}
						}
						DetRes.BestClass = nBestClass;
						DetRes.Score = fScore;
						vtResult.push_back(DetRes);
					}
				}
			}
		}
	}

	DoNMS(vtResult, fIOUThres, fScoreThres, nClass);

	return vtResult;
}