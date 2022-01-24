#pragma once
#include "Detection.h"
#include "math.h"

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

float Detection::CalculateIOU(DetectionResult box1, DetectionResult box2)
{
	int maxX = std::max(box1.x - (box1.w / 2), box2.x - (box2.w / 2));
	int maxY = std::max(box1.y - (box1.h / 2), box2.y - (box2.h / 2));
	int minX = std::min(box1.x + (box1.w / 2), box2.x + (box2.w / 2));
	int minY = std::min(box1.y + (box1.h / 2), box2.y + (box2.h / 2));
	int overlapWidth = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
	int overlapHeight = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
	int overlapArea = overlapWidth * overlapHeight;
	int box1Area = box1.h * box1.w;
	int box2Area = box2.h * box2.w;
	return float(overlapArea) / float(box1Area + box2Area - overlapArea);
}

void Detection::DoNMS(std::vector<DetectionResult>& detRes, float iouThresh, float scoreThresh, int clsNum)
{
	if (detRes.empty()) return;
	std::vector<std::vector<DetectionResult>> detResOfEachClass;
	for (int s = 0; s < clsNum; ++s)
	{
		std::vector<DetectionResult> tmp;
		for (std::vector<DetectionResult>::iterator it = detRes.begin(); it != detRes.end();)
		{
			if ((*it).BestClass == s)
			{
				tmp.push_back(*it);
				it = detRes.erase(it);
			}
			else it++;
		}
		ApplyScoreThreshold(tmp, scoreThresh);
		sort(tmp.begin(), tmp.end(), CompareScore);//sort the candidate boxes by confidence
		detResOfEachClass.push_back(tmp);
	}

	for (int s = 0; s < clsNum; ++s)
	{
		for (int i = 0; i < detResOfEachClass[s].size(); i++)
		{
			if (detResOfEachClass[s][i].Score > 0)
			{
				for (int j = i + 1; j < detResOfEachClass[s].size(); j++)
				{
					if (detResOfEachClass[s][j].Score > 0)
					{
						float iou = CalculateIOU(detResOfEachClass[s][i], detResOfEachClass[s][j]);//calculate the orthogonal ratio
						if (iou > iouThresh) detResOfEachClass[s][j].Score = 0;
					}
				}
			}
		}
		for (std::vector<DetectionResult>::iterator it = detResOfEachClass[s].begin(); it != detResOfEachClass[s].end(); ++it)
		{
			if ((*it).Score != 0) detRes.push_back(*it);
		}
	}
	return;
}

void Detection::ApplyScoreThreshold(std::vector<DetectionResult>& detRes, float scoreThresh)
{
	for (std::vector<DetectionResult>::iterator it = detRes.begin(); it != detRes.end();)
	{
		if ((*it).Objectness * (*it).Score < scoreThresh) it = detRes.erase(it);
		else it++;
	}
	return;
}

std::vector<std::vector<DetectionResult>> Detection::GetDetectionResults(float iouThresh, float scoreThresh)
{
	//Suppose there is only one output operation in detection tasks.
	int gridX = (int)mOutputDimsArr[0][1];
	int gridY = (int)mOutputDimsArr[0][2];
	int anchorNum = (int)mOutputDimsArr[0][3];
	int clsNum = ((int)mOutputDimsArr[0][4] - 6) / 2;

	int originalBatch = (int)TF_Dim(mOutputTensors[0][0], 0);

	std::vector<std::vector<DetectionResult>> result;

	for (int i = 0; i < mOutputTensors[0].size(); ++i)//Tensor iteration
	{
		int batch = (int)TF_Dim(mOutputTensors[0][i], 0);
		float *output = new float[batch * gridX * gridY * anchorNum * (clsNum * 2 + 6)];
		std::memcpy(output, TF_TensorData(mOutputTensors[0][i]), batch * gridX * gridY * anchorNum * (clsNum * 2 + 6) * sizeof(float));

		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)//Image in tensor iteration
		{
			std::vector<DetectionResult> tmpRes;
			int currImgIdx = i * originalBatch + imgIdx;
			for (int grdXIdx = 0; grdXIdx < gridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < gridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < anchorNum; ++ancIdx)
					{
						DetectionResult detRes;
						detRes.x = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							0];
						detRes.y = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							1];
						detRes.w = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							2];
						detRes.h = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							3];
						detRes.Objectness = output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							4];
						int bestCls = -1;
						float score = 0.;
						for (int clsIdx = 0; clsIdx < clsNum; ++clsIdx)
						{
							float currScore = output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6)
								+ grdXIdx * gridY * anchorNum * (clsNum * 2 + 6)
								+ grdYIdx * anchorNum * (clsNum * 2 + 6)
								+ ancIdx * (clsNum * 2 + 6)
								+ 5 + clsIdx];
							if (currScore >= score)
							{
								bestCls = clsIdx;
								score = currScore;
							}
						}
						detRes.BestClass = bestCls;
						detRes.Score = score;
						tmpRes.push_back(detRes);
					}
				}
			}
			DoNMS(tmpRes, iouThresh, scoreThresh, clsNum);
			result.push_back(tmpRes);
		}
	}

	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
		{
			TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
		}
		mOutputTensors[opsIdx].clear();
	}
	mOutputTensors.clear();

	return result;
}

bool Detection::GetDetectionResults(DetectionResult** detectionResultArr, int* boxNumArr, float iouThresh, float scoreThresh)
{
	//Suppose there is only one output operation in detection tasks.
	int gridX = (int)mOutputDimsArr[0][1];
	int gridY = (int)mOutputDimsArr[0][2];
	int anchorNum = (int)mOutputDimsArr[0][3];
	int clsNum = ((int)mOutputDimsArr[0][4] - 6) / 2;

	int originalBatch = (int)TF_Dim(mOutputTensors[0][0], 0);

	std::vector<std::vector<DetectionResult>> result;

	for (int i = 0; i < mOutputTensors[0].size(); ++i)//Tensor iteration
	{
		int batch = (int)TF_Dim(mOutputTensors[0][i], 0);
		float *output = new float[batch * gridX * gridY * anchorNum * (clsNum * 2 + 6)];
		std::memcpy(output, TF_TensorData(mOutputTensors[0][i]), batch * gridX * gridY * anchorNum * (clsNum * 2 + 6) * sizeof(float));

		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)//Image in tensor iteration
		{
			std::vector<DetectionResult> tmpRes;
			int currImgIdx = i * originalBatch + imgIdx;
			for (int grdXIdx = 0; grdXIdx < gridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < gridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < anchorNum; ++ancIdx)
					{
						DetectionResult detRes;
						detRes.x = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							0];
						detRes.y = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							1];
						detRes.w = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							2];
						detRes.h = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							3];
						detRes.Objectness = output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							4];
						int bestCls = -1;
						float score = 0.;
						for (int clsIdx = 0; clsIdx < clsNum; ++clsIdx)
						{
							float currScore = output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6)
								+ grdXIdx * gridY * anchorNum * (clsNum * 2 + 6)
								+ grdYIdx * anchorNum * (clsNum * 2 + 6)
								+ ancIdx * (clsNum * 2 + 6)
								+ 5 + clsIdx];
							if (currScore >= score)
							{
								bestCls = clsIdx;
								score = currScore;
							}
						}
						detRes.BestClass = bestCls;
						detRes.Score = score;
						tmpRes.push_back(detRes);
					}
				}
			}
			DoNMS(tmpRes, iouThresh, scoreThresh, clsNum);
			result.push_back(tmpRes);
		}
	}

	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
		{
			TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
		}
		mOutputTensors[opsIdx].clear();
	}
	mOutputTensors.clear();

	return true;
}

bool Detection::GetWholeImageDetectionResultsSingleOutput(DetectionResult* detResArr, int& boxNum, float iouThresh, float scoreThresh)
{
	//Suppose there is only one output operation in detection tasks.
	std::vector<DetectionResult> result;
	int gridX = (int)mOutputDimsArr[0][1];
	int gridY = (int)mOutputDimsArr[0][2];
	int anchorNum = (int)mOutputDimsArr[0][3];
	int clsNum = ((int)mOutputDimsArr[0][4] - 6) / 2;
	
	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - ((mCropSize.x - mOverlapSize.x) * (itX - 1)) > mCropSize.x) ++itX;
	if (mImageSize.y - ((mCropSize.y - mOverlapSize.y) * (itY - 1)) > mCropSize.y) ++itY;
	int currXIdx = 0;
	int currYIdx = 0;
	int currImgIdx = 0;

	int originalBatch = (int)TF_Dim(mOutputTensors[0][0], 0);
	
	for (int i = 0; i < mOutputTensors[0].size(); ++i)//Tensor iteration
	{
		int batch = (int)TF_Dim(mOutputTensors[0][i], 0);
		float *output = new float[batch * gridX * gridY * anchorNum * (clsNum * 2 + 6)];
		std::memcpy(output, TF_TensorData(mOutputTensors[0][i]), batch * gridX * gridY * anchorNum * (clsNum * 2 + 6) * sizeof(float));
		
		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)//Image in tensor iteration
		{
			currImgIdx = i * originalBatch + imgIdx;
			currXIdx = currImgIdx % itX;
			currYIdx = currImgIdx / itX;
			for (int grdXIdx = 0; grdXIdx < gridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < gridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < anchorNum; ++ancIdx)
					{
						int xOffset = (mCropSize.x - mOverlapSize.x) * currXIdx;
						int yOffset = (mCropSize.y - mOverlapSize.y) * currYIdx;
						if (xOffset + mCropSize.x > mImageSize.x) xOffset = mImageSize.x - mCropSize.x;
						if (yOffset + mCropSize.y > mImageSize.y) yOffset = mImageSize.y - mCropSize.y;

						DetectionResult detRes;
						detRes.x = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							0] +
							xOffset;
						detRes.y = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							1] +
							yOffset;
						detRes.w = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							2];
						detRes.h = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							3];
						detRes.Objectness = output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6) +
							grdXIdx * gridY * anchorNum * (clsNum * 2 + 6) +
							grdYIdx * anchorNum * (clsNum * 2 + 6) +
							ancIdx * (clsNum * 2 + 6) +
							4];
						int bestCls = -1;
						float score = 0.;
						for (int clsIdx = 0; clsIdx < clsNum; ++clsIdx)
						{
							float currScore = output[imgIdx * gridX * gridY * anchorNum * (clsNum * 2 + 6)
								+ grdXIdx * gridY * anchorNum * (clsNum * 2 + 6)
								+ grdYIdx * anchorNum * (clsNum * 2 + 6)
								+ ancIdx * (clsNum * 2 + 6)
								+ 5 + clsIdx];
							if (currScore >= score)
							{
								bestCls = clsIdx;
								score = currScore;
							}
						}
						detRes.BestClass = bestCls;
						detRes.Score = score;
						result.push_back(detRes);
					}
				}
			}
		}
	}

	DoNMS(result, iouThresh, scoreThresh, clsNum);

	boxNum = result.size();
	for (int i = 0; i < boxNum; ++i)
	{
		detResArr[i] = result[i];
	}

	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
		{
			TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
		}
		mOutputTensors[opsIdx].clear();
	}
	mOutputTensors.clear();

	return true;
}

bool Detection::GetWholeImageDetectionResultsDoubleOutput(DetectionResult* detResArr, int& boxNum, float iouThresh, float scoreThresh)
{

	//Suppose there is only one output operation in detection tasks.
	std::vector<DetectionResult> result;
	int batch = (int)mOutputDimsArr[1][0];
	int gridX = (int)mOutputDimsArr[1][1];
	int gridY = (int)mOutputDimsArr[1][2];
	int anchorNum = (int)mOutputDimsArr[1][3];
	int clsNum = (int)mOutputDimsArr[1][4] - 5;

	int itX = (int)(mImageSize.x / (mCropSize.x - mOverlapSize.x));
	int itY = (int)(mImageSize.y / (mCropSize.y - mOverlapSize.y));
	if (mImageSize.x - ((mCropSize.x - mOverlapSize.x) * (itX - 1)) > mCropSize.x) ++itX;
	if (mImageSize.y - ((mCropSize.y - mOverlapSize.y) * (itY - 1)) > mCropSize.y) ++itY;
	int currXIdx = 0;
	int currYIdx = 0;
	int currImgIdx = 0;

	for (int i = 0; i < mOutputTensors[1].size(); ++i)//Tensor iteration
	{
		float *output = new float[batch * gridX * gridY * anchorNum * (clsNum + 5)];
		std::memcpy(output, TF_TensorData(mOutputTensors[1][i]), batch * gridX * gridY * anchorNum * (clsNum + 5) * sizeof(float));

		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)//Image in tensor iteration
		{
			currImgIdx = i * batch + imgIdx;
			currXIdx = currImgIdx % itX;
			currYIdx = currImgIdx / itX;
			for (int grdXIdx = 0; grdXIdx < gridX; ++grdXIdx)
			{
				for (int grdYIdx = 0; grdYIdx < gridY; ++grdYIdx)
				{
					for (int ancIdx = 0; ancIdx < anchorNum; ++ancIdx)
					{
						int xOffset = (mCropSize.x - mOverlapSize.x) * currXIdx;
						int yOffset = (mCropSize.y - mOverlapSize.y) * currYIdx;
						if (xOffset + mCropSize.x > mImageSize.x) xOffset = mImageSize.x - mCropSize.x;
						if (yOffset + mCropSize.y > mImageSize.y) yOffset = mImageSize.y - mCropSize.y;

						DetectionResult detRes;
						detRes.x = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum + 5) +
							grdXIdx * gridY * anchorNum * (clsNum + 5) +
							grdYIdx * anchorNum * (clsNum + 5) +
							ancIdx * (clsNum + 5) +
							0] +
							xOffset;
						detRes.y = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum + 5) +
							grdXIdx * gridY * anchorNum * (clsNum + 5) +
							grdYIdx * anchorNum * (clsNum + 5) +
							ancIdx * (clsNum + 5) +
							1] +
							yOffset;
						detRes.w = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum + 5) +
							grdXIdx * gridY * anchorNum * (clsNum + 5) +
							grdYIdx * anchorNum * (clsNum + 5) +
							ancIdx * (clsNum + 5) +
							2];
						detRes.h = (int)output[imgIdx * gridX * gridY * anchorNum * (clsNum + 5) +
							grdXIdx * gridY * anchorNum * (clsNum + 5) +
							grdYIdx * anchorNum * (clsNum + 5) +
							ancIdx * (clsNum + 5) +
							3];
						detRes.Objectness = output[imgIdx * gridX * gridY * anchorNum * (clsNum + 5) +
							grdXIdx * gridY * anchorNum * (clsNum + 5) +
							grdYIdx * anchorNum * (clsNum + 5) +
							ancIdx * (clsNum + 5) +
							4];
						int bestCls = -1;
						float score = 0.;
						for (int clsIdx = 0; clsIdx < clsNum; ++clsIdx)
						{
							float currScore = output[imgIdx * gridX * gridY * anchorNum * (clsNum + 5)
								+ grdXIdx * gridY * anchorNum * (clsNum + 5)
								+ grdYIdx * anchorNum * (clsNum + 5)
								+ ancIdx * (clsNum + 5)
								+ 5 + clsIdx];
							if (currScore >= score)
							{
								bestCls = clsIdx;
								score = currScore;
							}
						}
						detRes.BestClass = bestCls;
						detRes.Score = score;
						result.push_back(detRes);
					}
				}
			}
		}
	}

	DoNMS(result, iouThresh, scoreThresh, clsNum);

	boxNum = result.size();
	for (int i = 0; i < boxNum; ++i)
	{
		detResArr[i] = result[i];
	}

	for (int opsIdx = 0; opsIdx < mInputOpNum; ++opsIdx)
	{
		for (int tensorIdx = 0; tensorIdx < mOutputTensors[opsIdx].size(); ++tensorIdx)
		{
			TF_DeleteTensor(mOutputTensors[opsIdx][tensorIdx]);
		}
		mOutputTensors[opsIdx].clear();
	}
	mOutputTensors.clear();

	return true;
}