#pragma once
#include "TFCore.h"


struct DetectionResult
{
	int x;
	int y;
	int w;
	int h;
	float Score;
	float Objectness;
	int BestClass;
};

class Detection : public TFCore
{
public:
	Detection();
	~Detection();

private:
	float CalculateIOU(DetectionResult, DetectionResult);
	void DoNMS(std::vector<DetectionResult>&, float);
};