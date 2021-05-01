#pragma once
#include "TFCore.h"

namespace TFTool
{
	class Detection : public TFCore
	{
	public:
		__declspec(dllexport) Detection();
		__declspec(dllexport) ~Detection();
	};

	class Yolo : public TFCore
	{
	public:
		__declspec(dllexport) Yolo();
		__declspec(dllexport) ~Yolo();
	};
}