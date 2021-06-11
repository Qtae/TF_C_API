#pragma once
#include "TFCore.h"


struct DetectionResult
{
	int x;
	int y;
	int w;
	int h;
	float Objectness;
	int BestClass;
	float Score;
};

class Detection : public TFCore
{
public:
	Detection();
	~Detection();
	std::vector<std::vector<DetectionResult>> GetDetectionResults(float fIOUThres = 0.5, float fScoreThres = 0.25);

private:
	float CalculateIOU(DetectionResult, DetectionResult);
	void DoNMS(std::vector<DetectionResult>&, float, float, int);
	void ApplyScoreThreshold(std::vector<DetectionResult>&, float);
};