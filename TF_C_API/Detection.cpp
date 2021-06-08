#include "Detection.h"

Detection::Detection()
{
}

Detection::~Detection()
{
}

bool CompareScore(DetectionResult x, DetectionResult y)
{
	if (x.Score > y.Score) return true;
	else return false;
}

float Detection::CalculateIOU(DetectionResult Box1, DetectionResult Box2)
{
	int MaxX = std::max(Box1.x - (Box1.w / 2), Box2.x - (Box2.w / 2));
	int MaxY = std::max(Box1.y - (Box1.h / 2), Box2.y - (Box2.h / 2));
	int MinX = std::min(Box1.x + (Box1.w / 2), Box2.x + (Box2.w / 2));
	int MinY = std::min(Box1.y + (Box1.h / 2), Box2.y + (Box2.h / 2));
	int OvelapWidth = ((MinX - MaxX + 1) > 0) ? (MinX - MaxX + 1) : 0;
	int OverlapHeight = ((MinY - MaxY + 1) > 0) ? (MinY - MaxY + 1) : 0;
	int OverlapArea = OvelapWidth * OverlapHeight;
	int Box1Area = Box1.h * Box1.w;
	int Box2Area = Box2.h * Box2.w;
	return float(OverlapArea) / float(Box1Area + Box2Area - OverlapArea);
}

void Detection::DoNMS(std::vector<DetectionResult> &vtDetRes, float fNMSThres)
{
	if (vtDetRes.empty()) return;
	sort(vtDetRes.begin(), vtDetRes.end(), CompareScore);//sort the candidate boxes by confidence
	for (int i = 0; i < vtDetRes.size(); i++)
	{
		if (vtDetRes[i].Score > 0)
		{
			for (int j = i + 1; j < vtDetRes.size(); j++)
			{
				if (vtDetRes[j].Score > 0)
				{
					float iou = CalculateIOU(vtDetRes[i], vtDetRes[j]);//calculate the orthogonal ratio
					if (iou > fNMSThres) vtDetRes[j].Score = 0;
				}
			}
		}
	}

	for (auto it = vtDetRes.begin(); it != vtDetRes.end();)
	{
		if ((*it).Score == 0) vtDetRes.erase(it);
		else it++;
	}
}