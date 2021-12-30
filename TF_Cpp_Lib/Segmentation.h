#pragma once
#include "TFCore.h"


class Segmentation : public TFCore
{
private:
	std::vector<std::vector<float*>> mOutputRes;
	std::vector<std::vector<int*>> mClassMask;

public:
	Segmentation();
	~Segmentation();
	std::vector<std::vector<float*>> GetOutput();
	std::vector<std::vector<int*>> GetWholeClsMask();
	std::vector<std::vector<int*>> GetBinaryMaskWithClsIndex(int);
	bool GetWholeImageSegmentationResults(unsigned char*, int);

private:
	bool FreeOutputMap();
};