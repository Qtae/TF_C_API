#pragma once
#include "Classification.h"
#include "Segmentation.h"
#include "Detection.h"
#include "TFTool.h"


using cv::Mat;
using cv::Vec3f;

int main()
{
	Mat Image1 = cv::imread("C:/Users/jhchoi/Desktop/VisionWorks/VisionWorks2/image/1_10_crop8.jpg", cv::IMREAD_COLOR);
	Mat Image2 = cv::imread("C:/Users/jhchoi/Desktop/VisionWorks/VisionWorks2/image/1_10_crop9.jpg", cv::IMREAD_COLOR);
	Mat Image3 = cv::imread("C:/Users/jhchoi/Desktop/VisionWorks/VisionWorks2/image/1_10_crop43.jpg", cv::IMREAD_COLOR);
	Mat Image4 = cv::imread("C:/Users/jhchoi/Desktop/VisionWorks/VisionWorks2/image/1_10_crop49.jpg", cv::IMREAD_COLOR);
	cv::cvtColor(Image1, Image1, cv::COLOR_BGR2RGB);
	cv::cvtColor(Image2, Image2, cv::COLOR_BGR2RGB);
	cv::cvtColor(Image3, Image3, cv::COLOR_BGR2RGB);
	cv::cvtColor(Image4, Image4, cv::COLOR_BGR2RGB);
	Mat Mat1(640, 640, CV_32FC(3));
	Mat Mat2(640, 640, CV_32FC(3));
	Mat Mat3(640, 640, CV_32FC(3));
	Mat Mat4(640, 640, CV_32FC(3));
	Image1.convertTo(Mat1, CV_32FC(3));
	Image2.convertTo(Mat2, CV_32FC(3));
	Image3.convertTo(Mat3, CV_32FC(3));
	Image4.convertTo(Mat4, CV_32FC(3));
	float** ImageSet = new float*[4];
	float* Image1f = new float[640 * 640 * 3];
	float* Image2f = new float[640 * 640 * 3];
	float* Image3f = new float[640 * 640 * 3];
	float* Image4f = new float[640 * 640 * 3];
	std::memcpy(Image1f, Mat1.data, 640 * 640 * 3 * sizeof(float));
	std::memcpy(Image2f, Mat2.data, 640 * 640 * 3 * sizeof(float));
	std::memcpy(Image3f, Mat3.data, 640 * 640 * 3 * sizeof(float));
	std::memcpy(Image4f, Mat4.data, 640 * 640 * 3 * sizeof(float));
	ImageSet[0] = Image1f;
	ImageSet[1] = Image2f;
	ImageSet[2] = Image3f;
	ImageSet[3] = Image4f;

	std::vector<const char*> vtInputOpNames;
	std::vector<const char*> vtOutputOpNames;
	vtInputOpNames.push_back("serving_default_input_1:0");
	vtOutputOpNames.push_back("StatefulPartitionedCall:1");
	const char* strModelPath = "C:/WISVision/StitchAI_Modelsc1_yolov4-tflite-train_tf-epoch100";

	TFTool::AI* AI = new TFTool::AI();
	AI->LoadModel(strModelPath, vtInputOpNames, vtOutputOpNames, 2);
	AI->Run(ImageSet, 2, true); //ImageData, Batch, bNormalize
	std::vector<std::vector<DetectionResult>> vtResult = AI->GetDetectionResults(0.3, 0.25);
}