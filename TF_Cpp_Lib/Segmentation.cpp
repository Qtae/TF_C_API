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
	if (!mOutputRes.empty())
		FreeOutputMap();

	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)//output operator °¹¼ö iteration
	{
		std::vector<float*> resultOp;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int outputSize = (int)TF_Dim(mOutputTensors[opsIdx][i], 1) * (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *outputImg = new float[batch * outputSize];
			std::memcpy(outputImg, TF_TensorData(mOutputTensors[opsIdx][i]), batch * outputSize * sizeof(float));
			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				float* batchOutputImg = new float[outputSize];
				std::memcpy(batchOutputImg, outputImg + imgIdx * outputSize, outputSize * sizeof(float));
				resultOp.push_back(batchOutputImg);
			}
			delete[] outputImg;
		}
		mOutputRes.push_back(resultOp);
	}
	return mOutputRes;
}

bool Segmentation::FreeOutputMap()
{
	if (!mOutputRes.empty())
	{
		for (int opsIdx = 0; opsIdx < mOutputRes.size(); ++opsIdx)
		{
			for (int nIdx = 0; nIdx < mOutputRes[opsIdx].size(); ++nIdx)
			{
				delete[] mOutputRes[opsIdx][nIdx];
			}
		}
	}
	return true;
}

std::vector<std::vector<int*>> Segmentation::GetWholeClsMask()
{
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)//output operator °¹¼ö iteration
	{
		std::vector<int*> clsMaskOp;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int outputSize = (int)TF_Dim(mOutputTensors[opsIdx][i], 1) * (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *outputImg = new float[batch * outputSize];
			std::memcpy(outputImg, TF_TensorData(mOutputTensors[opsIdx][i]), batch * outputSize * sizeof(float));
			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				float* batchOutputImg = new float[outputSize];
				std::memcpy(batchOutputImg, outputImg + imgIdx * outputSize, outputSize * sizeof(float));
			}
			delete[] outputImg;
		}
		mClassMask.push_back(clsMaskOp);
	}
	return mClassMask;
}

std::vector<std::vector<int*>> Segmentation::GetBinaryMaskWithClsIndex(int nClsIndex)
{
	for (int opsIdx = 0; opsIdx < mOutputOpNum; ++opsIdx)
	{
		std::vector<int*> clsMaskOp;
		for (int i = 0; i < mOutputTensors[opsIdx].size(); ++i)//Tensor iteration
		{
			int batch = (int)TF_Dim(mOutputTensors[opsIdx][i], 0);
			int outputSize = (int)TF_Dim(mOutputTensors[opsIdx][i], 1) * (int)TF_Dim(mOutputTensors[opsIdx][i], 1);
			float *outputImg = new float[batch * outputSize];
			std::memcpy(outputImg, TF_TensorData(mOutputTensors[opsIdx][i]), batch * outputSize * sizeof(float));
			for (int imgIdx = 0; imgIdx < batch; ++imgIdx)
			{
				int* batchOutputImg = new int[outputSize];
				for (int pixIdx = 0; pixIdx < outputSize; ++pixIdx)
				{
					if ((int)outputImg[imgIdx * outputSize + pixIdx] == nClsIndex)
						batchOutputImg[pixIdx] = 1;
					else
						batchOutputImg[pixIdx] = 0;
				}
				clsMaskOp.push_back(batchOutputImg);
			}
			delete[] outputImg;
		}
		mClassMask.push_back(clsMaskOp);
	}
	return mClassMask;
}

bool Segmentation::GetWholeImageSegmentationResults(unsigned char* outputImg, int clsNo)
{
	//Suppose there is only one output operation in detection tasks.
	int width = (int)mOutputDimsArr[0][1];
	int height = (int)mOutputDimsArr[0][2];
	int channel = (int)mOutputDimsArr[0][3];
	if (clsNo >= channel) return false;

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
		float *output = new float[batch * width * height * channel];
		std::memcpy(output, TF_TensorData(mOutputTensors[0][i]), batch * width * height * channel * sizeof(float));

		for (int imgIdx = 0; imgIdx < batch; ++imgIdx)//Image in tensor iteration
		{
			currImgIdx = i * originalBatch + imgIdx;
			currXIdx = currImgIdx % itX;
			currYIdx = currImgIdx / itX;
			int xOffset = (mCropSize.x - mOverlapSize.x) * currXIdx;
			int yOffset = (mCropSize.y - mOverlapSize.y) * currYIdx;
			if (xOffset + mCropSize.x > mImageSize.x) xOffset = mImageSize.x - mCropSize.x;
			if (yOffset + mCropSize.y > mImageSize.y) yOffset = mImageSize.y - mCropSize.y;

			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					outputImg[(yOffset + y) * mImageSize.x + xOffset + x] = output[imgIdx * height * width * channel + y * width * channel + x * channel + clsNo];
				}
			}
		}
		delete[] output;
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