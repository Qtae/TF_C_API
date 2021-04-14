#include "Classification.h"
#include "Segmentation.h"

int main()
{
	Classification *Cls = new Classification();

	std::vector<const char *> vtInputOpNames;
	std::vector<const char *> vtOutputOpNames;
	vtInputOpNames.push_back("serving_default_input_1");
	vtOutputOpNames.push_back("StatefulPartitionedCall");

	cv::Mat Image1 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/01.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image2 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/02.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image3 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/03.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image4 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/04.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image5 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/05.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image6 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/06.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image7 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/07.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image8 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/08.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image9 = cv::imread("D:/Public/qtkim/WIND2_ADC/DATA/Test/0.good/09.jpeg", cv::IMREAD_GRAYSCALE);
	cv::resize(Image1, Image1, cv::Size(224, 224));
	cv::resize(Image2, Image2, cv::Size(224, 224));
	cv::resize(Image3, Image3, cv::Size(224, 224));
	cv::resize(Image4, Image4, cv::Size(224, 224));
	cv::resize(Image5, Image5, cv::Size(224, 224));
	cv::resize(Image6, Image6, cv::Size(224, 224));
	cv::resize(Image7, Image7, cv::Size(224, 224));
	cv::resize(Image8, Image8, cv::Size(224, 224));
	cv::resize(Image9, Image9, cv::Size(224, 224));
	cv::Mat Mat1(224, 224, CV_32FC(1));
	cv::Mat Mat2(224, 224, CV_32FC(1));
	cv::Mat Mat3(224, 224, CV_32FC(1));
	cv::Mat Mat4(224, 224, CV_32FC(1));
	cv::Mat Mat5(224, 224, CV_32FC(1));
	cv::Mat Mat6(224, 224, CV_32FC(1));
	cv::Mat Mat7(224, 224, CV_32FC(1));
	cv::Mat Mat8(224, 224, CV_32FC(1));
	cv::Mat Mat9(224, 224, CV_32FC(1));
	Image1.convertTo(Mat1, CV_32FC(1));
	Image2.convertTo(Mat2, CV_32FC(1));
	Image3.convertTo(Mat3, CV_32FC(1));
	Image4.convertTo(Mat4, CV_32FC(1));
	Image5.convertTo(Mat5, CV_32FC(1));
	Image6.convertTo(Mat6, CV_32FC(1));
	Image7.convertTo(Mat7, CV_32FC(1));
	Image8.convertTo(Mat8, CV_32FC(1));
	Image9.convertTo(Mat9, CV_32FC(1));
	float ***ImageSet = new float**[9];
	float **Imageset1f = new float*[1];
	float **Imageset2f = new float*[1];
	float **Imageset3f = new float*[1];
	float **Imageset4f = new float*[1];
	float **Imageset5f = new float*[1];
	float **Imageset6f = new float*[1];
	float **Imageset7f = new float*[1];
	float **Imageset8f = new float*[1];
	float **Imageset9f = new float*[1];
	float *Image1f = new float[224 * 224];
	float *Image2f = new float[224 * 224];
	float *Image3f = new float[224 * 224];
	float *Image4f = new float[224 * 224];
	float *Image5f = new float[224 * 224];
	float *Image6f = new float[224 * 224];
	float *Image7f = new float[224 * 224];
	float *Image8f = new float[224 * 224];
	float *Image9f = new float[224 * 224];

	std::memcpy(Image1f, Mat1.data, 224 * 224 * sizeof(float));
	std::memcpy(Image2f, Mat2.data, 224 * 224 * sizeof(float));
	std::memcpy(Image3f, Mat3.data, 224 * 224 * sizeof(float));
	std::memcpy(Image4f, Mat4.data, 224 * 224 * sizeof(float));
	std::memcpy(Image5f, Mat5.data, 224 * 224 * sizeof(float));
	std::memcpy(Image6f, Mat6.data, 224 * 224 * sizeof(float));
	std::memcpy(Image7f, Mat7.data, 224 * 224 * sizeof(float));
	std::memcpy(Image8f, Mat8.data, 224 * 224 * sizeof(float));
	std::memcpy(Image9f, Mat9.data, 224 * 224 * sizeof(float));

	Imageset1f[0] = Image1f;
	Imageset2f[0] = Image2f;
	Imageset3f[0] = Image3f;
	Imageset4f[0] = Image4f;
	Imageset5f[0] = Image5f;
	Imageset6f[0] = Image6f;
	Imageset7f[0] = Image7f;
	Imageset8f[0] = Image8f;
	Imageset9f[0] = Image9f;

	ImageSet[0] = Imageset1f;
	ImageSet[1] = Imageset2f;
	ImageSet[2] = Imageset3f;
	ImageSet[3] = Imageset4f;
	ImageSet[4] = Imageset5f;
	ImageSet[5] = Imageset6f;
	ImageSet[6] = Imageset7f;
	ImageSet[7] = Imageset8f;
	ImageSet[8] = Imageset9f;

	std::vector<std::vector<std::vector<float>>> vtResult;

	//////////////////////////////////////////////////////////////////////////////////////////////
	Cls->LoadModel("D:/Public/qtkim/WIND2_ADC/saved_model_adc/model_1", vtInputOpNames, vtOutputOpNames);
	Cls->Run(ImageSet, 9, 4); //ImageData, ImageDataNum, Batch
	vtResult = Cls->GetResult();
	Cls->FreeModel();
}