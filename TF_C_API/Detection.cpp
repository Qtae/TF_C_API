#include "Detection.h"

Detection::Detection()
{
}

Detection::~Detection()
{
}

bool CompareScore(DetectionResult x, DetectionResult y)
{
	if (x.Score > y.Score) return true;
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

void Detection::DoNMS(std::vector<DetectionResult> &vtDetRes, float fNMSThres)
{
	if (vtDetRes.empty()) return;
	sort(vtDetRes.begin(), vtDetRes.end(), CompareScore);//sort the candidate boxes by confidence
	for (int i = 0; i < vtDetRes.size(); i++)
	{
		if (vtDetRes[i].Score > 0)
		{
			for (int j = i + 1; j < vtDetRes.size(); j++)
			{
				if (vtDetRes[j].Score > 0)
				{
					float iou = CalculateIOU(vtDetRes[i], vtDetRes[j]);//calculate the orthogonal ratio
					if (iou < fNMSThres) vtDetRes[j].Score = 0;
				}
			}
		}
	}

	for (std::vector<DetectionResult>::iterator it = vtDetRes.begin(); it != vtDetRes.end();)
	{
		if ((*it).Score == 0) it = vtDetRes.erase(it);
		else it++;
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

std::vector<DetectionResult> Detection::GetDetectionResults(float fNMSThres, float fScoreThres)
{
	//Suppose there is only one output operation in detection tasks.
	std::vector<DetectionResult> vtResult;
	int nBatch = (int)m_OutputDims[0][0];
	int nGridX = (int)m_OutputDims[0][1];
	int nGridY = (int)m_OutputDims[0][2];
	int nAnchor = (int)m_OutputDims[0][3];
	int nClass = (int)m_OutputDims[0][4] - 5;

	for (int i = 0; i < m_vtOutputTensors[0].size(); ++i)//Tensor iteration
	{
		float *output = new float[nBatch * nGridX * nGridY * nAnchor * (nClass + 5)];
		std::memcpy(output, TF_TensorData(m_vtOutputTensors[0][i]), nBatch * nGridX * nGridY * nAnchor * (nClass + 5) * sizeof(float));

		for (int batchIdx = 0; batchIdx < nBatch; ++batchIdx)
		{
			for (int grdXIdx = 0; grdXIdx < nGridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < nGridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < nAnchor; ++ancIdx)
					{
						DetectionResult DetRes;
						DetRes.x = output[batchIdx * nGridX * nGridY * nAnchor * (nClass + 5)
										+ grdXIdx * nGridY * nAnchor * (nClass + 5)
										+ grdYIdx * nAnchor * (nClass + 5)
										+ ancIdx * (nClass + 5)
										+ 0];
						DetRes.y = output[batchIdx * nGridX * nGridY * nAnchor * (nClass + 5)
										+ grdXIdx * nGridY * nAnchor * (nClass + 5)
										+ grdYIdx * nAnchor * (nClass + 5)
										+ ancIdx * (nClass + 5)
										+ 1];
						DetRes.w = output[batchIdx * nGridX * nGridY * nAnchor * (nClass + 5)
										+ grdXIdx * nGridY * nAnchor * (nClass + 5)
										+ grdYIdx * nAnchor * (nClass + 5)
										+ ancIdx * (nClass + 5)
										+ 2];
						DetRes.h = output[batchIdx * nGridX * nGridY * nAnchor * (nClass + 5)
										+ grdXIdx * nGridY * nAnchor * (nClass + 5)
										+ grdYIdx * nAnchor * (nClass + 5)
										+ ancIdx * (nClass + 5)
										+ 3];
						DetRes.Objectness = output[batchIdx * nGridX * nGridY * nAnchor * (nClass + 5)
												 + grdXIdx * nGridY * nAnchor * (nClass + 5)
												 + grdYIdx * nAnchor * (nClass + 5)
												 + ancIdx * (nClass + 5)
												 + 4];
						int nBestClass = -1;
						float fScore = 0.;
						for (int clsIdx = 0; clsIdx < nClass; ++clsIdx)
						{
							float fCurrScore = output[batchIdx * nGridX * nGridY * nAnchor * (nClass + 5)
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

	ApplyScoreThreshold(vtResult, fScoreThres);
	DoNMS(vtResult, fNMSThres);

	return vtResult;
}