#pragma once
#include <vector>
#include "TFResultStructure.h"


class Classification;
class Segmentation;
class Detection;

template class __declspec(dllexport) std::vector<DetectionResult>;

namespace TFTool
{
	class AI
	{
	public:
		//Constructor of AI Instance.
		__declspec(dllexport) AI();

		//Destructor
		__declspec(dllexport) ~AI();

		//Load Saved Model Format Directory
		//	ModelPath : 
		//	vtInputOpNames : Vector of input names
		//					 (Format : {input operation name}:{index} )
		//	vtOutputOpNames : Vector of output names
		//					 (Format : {output operation name}:{index} )
		//	nTaskType : Task type
		//		0 : Classification
		//		1 : Segmentation
		//		2 : Detection
		__declspec(dllexport) bool LoadModel(const char* ModelPath,
			std::vector<const char*> &vtInputOpNames, std::vector<const char*>& vtOutputOpNames,
			int nTaskType);

		//Run Session
		//	pImageSet[j][k] : Array of input image(Float)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float** pImageSet, bool bNormalize = false);

		//Run Session
		//	pImageSet[i][j][k] : Array of input image(Float)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float*** pImageSet, bool bNormalize = false);

		//Run Session
		//	pImageSet[j][k] : Array of input image(Byte)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char** pImageSet, bool bNormalize = false);

		//Run Session
		//	pImageSet[i][j][k] : Array of input image(Byte)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char*** pImageSet, bool bNormalize = false);

		//Run Session
		//	pImageSet[i][j][k] : Array of input image(Float)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	nBatch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float*** pImageSet, int nBatch, bool bNormalize = false);

		//Run Session
		//	pImageSet[j][k] : Array of input image(Float)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	nBatch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float** pImageSet, int nBatch, bool bNormalize = false);

		//Run Session
		//	pImageSet[i][j][k] : Array of input image(Byte)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	nBatch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char*** pImageSet, int nBatch, bool bNormalize = false);

		//Run Session
		//	pImageSet[j][k] : Array of input image(Byte)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	nBatch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char** pImageSet, int nBatch, bool bNormalize = false);

		//Run Session
		//	pImageSet :Image Byte Array
		//	nImageSizeX, nImageSizeY : Image size
		//	nCropSizeX, nCropSizeY : Crop size
		//	nOverlapSizeX, nOverlapSizeY : Overlap size
		//	nBatch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		//	bConvertGrayToColor : Convert Grayscale input image to color image.
		__declspec(dllexport) bool Run(unsigned char** ppImage, int nImageSizeX, int nImageSizeY,
			int nCropSizeX, int nCropSizeY, int nOverlapSizeX, int nOverlapSizeY, int nBuffPosX, int nBuffPosY,
			int nBatch, bool bNormalize = false, bool bConvertGrayToColor = false);

		//Free memory
		__declspec(dllexport) bool FreeModel();

		//Returns classification result
		__declspec(dllexport) std::vector<std::vector<std::vector<float>>> GetClassificationResults();

		//Returns classification result
		__declspec(dllexport) std::vector<std::vector<int>> GetClassificationResults(float fSoftmxThresh = 0);

		//Returns segmentation result
		__declspec(dllexport) std::vector<std::vector<float*>> GetSegmentationResults();

		//Returns detection result
		__declspec(dllexport) std::vector<std::vector<DetectionResult>> GetDetectionResults(float fIOUThres = 0.5, float fScoreThres = 0.25);

		//Return whole image (cropped on run() function) detection result
		__declspec(dllexport) bool GetWholeImageDetectionResults(DetectionResult*, int&, float fIOUThres = 0.5, float fScoreThres = 0.25);

		//Returns true when model is loaded
		__declspec(dllexport) bool IsModelLoaded();

		//Returns input/output dimensions
		__declspec(dllexport) long long** GetInputDims();
		__declspec(dllexport) long long** GetOutputDims();

	private:
		int m_nTaskType = -1;

		Classification *pClassification;
		Segmentation* pSegmentation;
		Detection* pDetection;
	};
}