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
	private:
		int mTaskType = -1;

		Classification* mClassification;
		Segmentation* mSegmentation;
		Detection* mDetection;

	public:
		//Constructor of AI Instance.
		__declspec(dllexport) AI();

		//Destructor
		__declspec(dllexport) ~AI();

		//Load Saved Model Format Directory
		//	modelPath : 
		//	inputOpNames : Vector of input names
		//					 (Format : {input operation name}:{index} )
		//	outputOpNames : Vector of output names
		//					 (Format : {output operation name}:{index} )
		//	taskType : Task type
		//		0 : Classification
		//		1 : Segmentation
		//		2 : Detection
		__declspec(dllexport) bool LoadModel(const char* modelPath,
			std::vector<const char*> &inputOpNames, std::vector<const char*>& outputOpNames,
			int taskType);

		//Run Session
		//	inputImgArr[j][k] : Array of input image(Float)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float** inputImgArr, bool bNormalize = false);

		//Run Session
		//	inputImgArr[i][j][k] : Array of input image(Float)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float*** inputImgArr, bool bNormalize = false);

		//Run Session
		//	inputImgArr[j][k] : Array of input image(Byte)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char** inputImgArr, bool bNormalize = false);

		//Run Session
		//	inputImgArr[i][j][k] : Array of input image(Byte)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char*** inputImgArr, bool bNormalize = false);

		//Run Session
		//	inputImg : Whole input image
		//	imgSizeX, imgSizeY : Image size
		//	cropSizeX, cropSizeY : Crop size
		//	overlapSizeX, overlapSizeY : Overlap size
		//  buffPosX, buffPosY : Buffer position of image origin.
		//	bNormalize : Divide pixel value by 255 when true.
		//	bConvertGrayToColor : Convert Grayscale input image to color image.
		__declspec(dllexport) bool Run(unsigned char** inputImg, int imgSizeX, int imgSizeY,
			int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY,
			bool bNormalize = false, bool bConvertGrayToColor = false);

		//Run Session
		//	inputImgArr[i][j][k] : Array of input image(Float)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	batch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float*** inputImgArr, int batch, bool bNormalize = false);

		//Run Session
		//	inputImgArr[j][k] : Array of input image(Float)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	batch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(float** inputImgArr, int batch, bool bNormalize = false);

		//Run Session
		//	inputImgArr[i][j][k] : Array of input image(Byte)
		//		i : input operator index
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	batch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char*** inputImgArr, int batch, bool bNormalize = false);

		//Run Session
		//	inputImgArr[j][k] : Array of input image(Byte)
		//		j : image index
		//		k : pixel index (x, y, channel)
		//	batch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		__declspec(dllexport) bool Run(unsigned char** inputImgArr, int batch, bool bNormalize = false);

		//Run Session
		//	inputImg : Whole input image
		//	imgSizeX, imgSizeY : Image size
		//	cropSizeX, cropSizeY : Crop size
		//	overlapSizeX, overlapSizeY : Overlap size
		//  buffPosX, buffPosY : Buffer position of image origin.
		//	batch : Image batch size
		//	bNormalize : Divide pixel value by 255 when true.
		//	bConvertGrayToColor : Convert Grayscale input image to color image.
		__declspec(dllexport) bool Run(unsigned char** inputImg, int imgSizeX, int imgSizeY,
			int cropSizeX, int cropSizeY, int overlapSizeX, int overlapSizeY, int buffPosX, int buffPosY,
			int batch, bool bNormalize = false, bool bConvertGrayToColor = false);

		//Free memory
		__declspec(dllexport) bool FreeModel();

		//Returns classification result
		//pClassificationResultArray
		__declspec(dllexport) bool GetClassificationResults(float*** classificationResultArr);

		//Returns classification result
		__declspec(dllexport) std::vector<std::vector<int>> GetClassificationResults(float softmxThresh = 0);

		//Return SortMX value
		__declspec(dllexport) std::vector<std::vector<std::vector<float>>> GetClassificationSoftMXResults();

		//Returns segmentation result
		__declspec(dllexport) std::vector<std::vector<float*>> GetSegmentationResults();

		//Returns detection result
		__declspec(dllexport) std::vector<std::vector<DetectionResult>> GetDetectionResults(float iouThresh = 0.5, float scoreThresh = 0.3);

		//Return whole image (cropped on run() function) detection result
		__declspec(dllexport) bool GetWholeImageDetectionResults(DetectionResult* detectionResultArr, int& boxNum, float iouThresh = 0.5, float scoreThresh = 0.3);

		//Return whole image (cropped on run() function) segmentation result
		__declspec(dllexport) bool GetWholeImageSegmentationResults(unsigned char* outputImg, int clsNo);

		//Returns true when model is loaded
		__declspec(dllexport) bool IsModelLoaded();

		//Returns input/output dimensions
		__declspec(dllexport) long long** GetInputDims();
		__declspec(dllexport) long long** GetOutputDims();
	};
}