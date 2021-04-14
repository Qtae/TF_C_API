#pragma once
#include "TFCore.h"


class Classification : public TFCore
{
public:
	Classification();
	~Classification();
	std::vector<std::vector<std::vector<float>>> GetResult();
};