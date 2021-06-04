#include "Classification.h"
#include "Segmentation.h"

int main()
{
	/*
	TFTool::Classification *Cls = new TFTool::Classification();

	std::vector<const char *> vtInputOpNames;
	std::vector<const char *> vtOutputOpNames;
	vtInputOpNames.push_back("serving_default_input_1");
	vtOutputOpNames.push_back("StatefulPartitionedCall");

	cv::Mat *tt = new cv::Mat();

	cv::Mat Image1 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/01.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image2 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/02.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image3 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/03.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image4 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/04.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image5 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/05.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image6 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/06.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image7 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/07.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image8 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/08.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat Image9 = cv::imread("D:/Work/03_WIND2_REVIEW/DATA/Test/0.good/09.jpeg", cv::IMREAD_GRAYSCALE);
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

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int n;
	std::cout << "Started. Press Enter" << std::endl;
	std::cin >> n;
	Cls->LoadModel("D:/Work/01_TF_C/saved_model_adc/model_1", vtInputOpNames, vtOutputOpNames);
	Cls->Run(ImageSet, 9, 4); //ImageData, ImageDataNum, Batch
	vtResult = Cls->GetResult();
	int m;
	std::cout << "Run() Ended. Press Enter" << std::endl;
	std::cin >> m;
	Cls->FreeModel();
	int k;
	std::cout << "FreeModel() Ended. Press Enter" << std::endl;
	std::cin >> k;
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	*/

	/*
	TFTool::Segmentation* Seg = new TFTool::Segmentation();

	std::vector<const char*> vtInputOpNames;
	std::vector<const char*> vtOutputOpNames;
	vtInputOpNames.push_back("serving_default_input_1");
	vtOutputOpNames.push_back("StatefulPartitionedCall");

	cv::Mat Image1 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0441.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image2 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0442.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image3 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0443.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image4 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0444.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image5 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0445.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image6 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0446.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image7 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0447.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image8 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0448.png", cv::IMREAD_GRAYSCALE);
	cv::Mat Image9 = cv::imread("D:/Work/01_TF_C/Data/CornerImage/IMAGE/0449.png", cv::IMREAD_GRAYSCALE);

	cv::Mat Mat1(64, 64, CV_32FC(1));
	cv::Mat Mat2(64, 64, CV_32FC(1));
	cv::Mat Mat3(64, 64, CV_32FC(1));
	cv::Mat Mat4(64, 64, CV_32FC(1));
	cv::Mat Mat5(64, 64, CV_32FC(1));
	cv::Mat Mat6(64, 64, CV_32FC(1));
	cv::Mat Mat7(64, 64, CV_32FC(1));
	cv::Mat Mat8(64, 64, CV_32FC(1));
	cv::Mat Mat9(64, 64, CV_32FC(1));
	Image1.convertTo(Mat1, CV_32FC(1));
	Image2.convertTo(Mat2, CV_32FC(1));
	Image3.convertTo(Mat3, CV_32FC(1));
	Image4.convertTo(Mat4, CV_32FC(1));
	Image5.convertTo(Mat5, CV_32FC(1));
	Image6.convertTo(Mat6, CV_32FC(1));
	Image7.convertTo(Mat7, CV_32FC(1));
	Image8.convertTo(Mat8, CV_32FC(1));
	Image9.convertTo(Mat9, CV_32FC(1));
	float*** ImageSet = new float** [9];
	float** Imageset1f = new float* [1];
	float** Imageset2f = new float* [1];
	float** Imageset3f = new float* [1];
	float** Imageset4f = new float* [1];
	float** Imageset5f = new float* [1];
	float** Imageset6f = new float* [1];
	float** Imageset7f = new float* [1];
	float** Imageset8f = new float* [1];
	float** Imageset9f = new float* [1];
	float* Image1f = new float[64 * 64];
	float* Image2f = new float[64 * 64];
	float* Image3f = new float[64 * 64];
	float* Image4f = new float[64 * 64];
	float* Image5f = new float[64 * 64];
	float* Image6f = new float[64 * 64];
	float* Image7f = new float[64 * 64];
	float* Image8f = new float[64 * 64];
	float* Image9f = new float[64 * 64];
	std::memcpy(Image1f, Mat1.data, 64 * 64 * sizeof(float));
	std::memcpy(Image2f, Mat2.data, 64 * 64 * sizeof(float));
	std::memcpy(Image3f, Mat3.data, 64 * 64 * sizeof(float));
	std::memcpy(Image4f, Mat4.data, 64 * 64 * sizeof(float));
	std::memcpy(Image5f, Mat5.data, 64 * 64 * sizeof(float));
	std::memcpy(Image6f, Mat6.data, 64 * 64 * sizeof(float));
	std::memcpy(Image7f, Mat7.data, 64 * 64 * sizeof(float));
	std::memcpy(Image8f, Mat8.data, 64 * 64 * sizeof(float));
	std::memcpy(Image9f, Mat9.data, 64 * 64 * sizeof(float));
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

	unsigned char*** ImageSet = new unsigned char**[9];
	unsigned char** Imageset1b = new unsigned char*[1];
	unsigned char** Imageset2b = new unsigned char*[1];
	unsigned char** Imageset3b = new unsigned char*[1];
	unsigned char** Imageset4b = new unsigned char*[1];
	unsigned char** Imageset5b = new unsigned char*[1];
	unsigned char** Imageset6b = new unsigned char*[1];
	unsigned char** Imageset7b = new unsigned char*[1];
	unsigned char** Imageset8b = new unsigned char*[1];
	unsigned char** Imageset9b = new unsigned char*[1];
	unsigned char* Image1b = new unsigned char[64 * 64];
	unsigned char* Image2b = new unsigned char[64 * 64];
	unsigned char* Image3b = new unsigned char[64 * 64];
	unsigned char* Image4b = new unsigned char[64 * 64];
	unsigned char* Image5b = new unsigned char[64 * 64];
	unsigned char* Image6b = new unsigned char[64 * 64];
	unsigned char* Image7b = new unsigned char[64 * 64];
	unsigned char* Image8b = new unsigned char[64 * 64];
	unsigned char* Image9b = new unsigned char[64 * 64];
	std::memcpy(Image1b, Image1.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image2b, Image2.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image3b, Image3.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image4b, Image4.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image5b, Image5.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image6b, Image6.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image7b, Image7.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image8b, Image8.data, 64 * 64 * sizeof(unsigned char));
	std::memcpy(Image9b, Image9.data, 64 * 64 * sizeof(unsigned char));
	std::vector<std::vector<float*>> vtResult;
	Imageset1b[0] = Image1b;
	Imageset2b[0] = Image2b;
	Imageset3b[0] = Image3b;
	Imageset4b[0] = Image4b;
	Imageset5b[0] = Image5b;
	Imageset6b[0] = Image6b;
	Imageset7b[0] = Image7b;
	Imageset8b[0] = Image8b;
	Imageset9b[0] = Image9b;
	ImageSet[0] = Imageset1b;
	ImageSet[1] = Imageset2b;
	ImageSet[2] = Imageset3b;
	ImageSet[3] = Imageset4b;
	ImageSet[4] = Imageset5b;
	ImageSet[5] = Imageset6b;
	ImageSet[6] = Imageset7b;
	ImageSet[7] = Imageset8b;
	ImageSet[8] = Imageset9b;

	Seg->LoadModel("D:/Work/01_TF_C/segmentation_model", vtInputOpNames, vtOutputOpNames);
	Seg->Run(ImageSet, 4); //ImageData, Batch
	vtResult = Seg->GetOutput();
	int i = 0;
	for (std::vector<float*>::iterator it = vtResult[0].begin(); it != vtResult[0].end(); ++it)
	{
		float* imagef = *it;
		cv::Mat MatImage(64, 64, CV_32FC(1));
		std::memcpy(MatImage.data, imagef, 64 * 64 * sizeof(float));
		cv::imshow("img", MatImage);
		//cv::imwrite("D:/Work/01_TF_C/99.jpeg", MatImage);
		cv::waitKey();
	}

	Seg->FreeModel();
	*/

	TFTool::Segmentation* Seg = new TFTool::Segmentation();
}