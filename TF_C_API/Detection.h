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
	std::vector<DetectionResult> GetDetectionResults(float fNMSThres = 0.5, float fScoreThres = 0.25);

private:
	float CalculateIOU(DetectionResult, DetectionResult);
	void DoNMS(std::vector<DetectionResult>&, float);
	void ApplyScoreThreshold(std::vector<DetectionResult>&, float);
};