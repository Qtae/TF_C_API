#pragma once
#include "Classification.h"
#include "Segmentation.h"
#include "Detection.h"

namespace TFTool
{
	typedef struct ModelInfo
	{
		int nModelType;
		const char* ModelName;
		std::vector<const char*> vtInputOpNames;
		std::vector<const char*> vtOutputOpNames;
	}ModelInfo;

	bool LoadModel(const char*, std::vector<const char*>&, std::vector<const char*>&, int);

	bool Run(float**, bool bNormalize = false);
	bool Run(float***, bool bNormalize = false);
	bool Run(unsigned char**, bool bNormalize = false);
	bool Run(unsigned char***, bool bNormalize = false);

	bool Run(float***, int, bool bNormalize = false);
	bool Run(float**, int, bool bNormalize = false);
	bool Run(unsigned char***, int, bool bNormalize = false);
	bool Run(unsigned char**, int, bool bNormalize = false);

	//VisionWorks image input format, has only one input operator
	bool Run(unsigned char**, CPoint, CPoint, int, bool bNormalize = false); std::vector<std::vector<std::vector<float>>> GetOutput();
	
	bool FreeModel();

	std::vector<std::vector<float>> GetOutputByOpIndex(int nOutputOpIndex = 0);
	std::vector<std::vector<int>> GetPredCls(float);
	std::vector<int> GetPredClsByOpIndex(float, int nOutputOpIndex = 0);
	void GetPredClsAndSftmx(std::vector<std::vector<int>>&, std::vector<std::vector<float>>&, float);
	void GetPredClsAndSftmxByOpIndex(std::vector<int>&, std::vector<float>&, float, int nOutputOpIndex = 0);
}