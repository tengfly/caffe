#pragma once

namespace caffe
{

	#define MAX_OBJ_COUNT 100
	#define MAX_LABEL_LEN 20
	#pragma pack(1)
	struct DetectedObject
	{
		char label[MAX_LABEL_LEN];
		int labelid;
		float score;
		int xmin;
		int ymin;
		int xmax;
		int ymax;
	};

	#pragma pack(1)
	struct DetectedObjects
	{
		int number;
		DetectedObject* objects;
	};

	class ssd_detector
	{
	public:
		void detect_init(const char* model_path, const int gpu_id);
		const DetectedObjects* detect(unsigned char* data, int height, int width, float confidence_threshold);
		void detect_stop(void);
	private:
		void* myssd;
		DetectedObjects* objects;
		int gpuid = -1;
		bool initialized = false;
		void* label_dic;
	};
}