#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Detection : public TFCore
	{
	public:
		Detection();
		~Detection();
	};

	class Yolo : public TFCore
	{
	public:
		Yolo();
		~Yolo();
	};
}