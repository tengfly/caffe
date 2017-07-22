// Referenced the demo code for using a SSD model to do detection.
// This is a demo code for using a SSD model to do detection.
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <caffe/prediction.hpp>
#include <windows.h>
#include <gflags/gflags.h>
#include <caffe/ssd_util/io.hpp>
#include <caffe/proto/labelmap.pb.h>
#include <google/protobuf/message.h>
#include <caffe/ssd_detection.hpp>
//#include <caffe/include_symbols.hpp>

using namespace std;
using namespace cv;
using cv::Mat;
using namespace objectDetect;
using namespace ssdnative;

namespace caffe {
	long long milliseconds_now() {
		static LARGE_INTEGER s_frequency;
		static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
		if (s_use_qpc) {
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			return (1000LL * now.QuadPart) / s_frequency.QuadPart;
		}
		else {
			return GetTickCount();
		}
	}

	void OutputDebugPrintf(const char * strOutputString, ...)
	{
		char strBuffer[4096] = { 0 };
		va_list vlArgs;
		va_start(vlArgs, strOutputString);
		//_vsnprintf(strBuffer, sizeof(strBuffer) - 1, strOutputString, vlArgs);
		vsprintf_s(strBuffer, strOutputString, vlArgs);
		va_end(vlArgs);
		OutputDebugStringA(strBuffer);
	}


	//std::vector<std::string> label_list = { "background", "accordion", "airplane", "ant", "antelope", "apple", "armadillo", "artichoke", "axe", "baby_bed", "backpack", "bagel", "balance_beam", "banana", "band_aid", "banjo", "baseball", "basketball", "bathing_cap", "beaker", "bear", "bee", "bell_pepper", "bench", "bicycle", "binder", "bird", "bookshelf", "bow", "bow_tie", "bowl", "brassiere", "burrito", "bus", "butterfly", "camel", "can_opener", "car", "cart", "cattle", "cello", "centipede", "chain_saw", "chair", "chime", "cocktail_shaker", "coffee_maker", "computer_keyboard", "computer_mouse", "corkscrew", "cream", "croquet_ball", "crutch", "cucumber", "cup_or_mug", "diaper", "digital_clock", "dishwasher", "dog", "domestic_cat", "dragonfly", "drum", "dumbbell", "electric_fan", "elephant", "face_powder", "fig", "filing_cabinet", "flower_pot", "flute", "fox", "french_horn", "frog", "frying_pan", "giant_panda", "goldfish", "golf_ball", "golfcart", "guacamole", "guitar", "hair_dryer", "hair_spray", "hamburger", "hammer", "hamster", "harmonica", "harp", "hat_with_a_wide_brim", "head_cabbage", "helmet", "hippopotamus", "horizontal_bar", "horse", "hotdog", "iPod", "isopod", "jellyfish", "koala_bear", "ladle", "ladybug", "lamp", "laptop", "lemon", "lion", "lipstick", "lizard", "lobster", "maillot", "maraca", "microphone", "microwave", "milk_can", "miniskirt", "monkey", "motorcycle", "mushroom", "nail", "neck_brace", "oboe", "orange", "otter", "pencil_box", "pencil_sharpener", "perfume", "person", "piano", "pineapple", "ping-pong_ball", "pitcher", "pizza", "plastic_bag", "plate_rack", "pomegranate", "popsicle", "porcupine", "power_drill", "pretzel", "printer", "puck", "punching_bag", "purse", "rabbit", "racket", "ray", "red_panda", "refrigerator", "remote_control", "rubber_eraser", "rugby_ball", "ruler", "salt_or_pepper_shaker", "saxophone", "scorpion", "screwdriver", "seal", "sheep", "ski", "skunk", "snail", "snake", "snowmobile", "snowplow", "soap_dispenser", "soccer_ball", "sofa", "spatula", "squirrel", "starfish", "stethoscope", "stove", "strainer", "strawberry", "stretcher", "sunglasses", "swimming_trunks", "swine", "syringe", "table", "tape_player", "tennis_ball", "tick", "tie", "tiger", "toaster", "traffic_light", "train", "trombone", "trumpet", "turtle", "tv_or_monitor", "unicycle", "vacuum", "violin", "volleyball", "waffle_iron", "washer", "water_bottle", "watercraft", "whale", "wine_bottle", "zebra" };
	//std::vector<std::string> label_list = { "background", "person", "backpack", "handbag", "tie", "suitcase", "bottle", "wine glass", "cup", "cell phone"};

	// Iterates though all people in the AddressBook and prints info about them.
	void ListItem(const LabelMap& label_map, map<int, string>& label_dic) {
		for (int i = 0; i < label_map.item_size(); i++) {
			const Item& item = label_map.item(i);

			OutputDebugPrintf("Item Label: %d, Name: %s", item.label(), item.display_name());
			label_dic[item.label()] = item.display_name();
		}
	}

	void ssd_detector::detect_init(const char* model_path, const int gpu_id)
	{
		string path = model_path;
		gpuid = gpu_id;
		const string model_file = path + "/deploy.prototxt";
		const string weights_file = path + "/model.caffemodel";
		const string labelmap_file = path + "/labelmap.prototxt";
		const string mean_file = "";// path + "/mean.binaryproto";
		const string label_file = "";
		//const string& mean_value = "104, 117, 123";		
		myssd = (void *)(new Predictor(model_file, weights_file, mean_file, label_file));
		objects = new DetectedObjects();
		objects->objects = new DetectedObject[MAX_OBJ_COUNT];

		// Verify that the version of the library that we linked against is
		// compatible with the version of the headers we compiled against.
		if (!labelmap_file.empty())
		{
			OutputDebugPrintf("Load label map.");

			GOOGLE_PROTOBUF_VERIFY_VERSION;
			LabelMap label_map;
			label_dic = (void *)(new map<int, string>());
			if (ReadProtoFromTextFile(labelmap_file, &label_map)) {
				ListItem(label_map, *((map<int, string> *)label_dic));
			}
			else {
				OutputDebugPrintf("File not found!");
			}
			// Optional:  Delete all global objects allocated by libprotobuf.
			google::protobuf::ShutdownProtobufLibrary();
		}
		//for (int i = 0; i < MAX_OBJ_COUNT; i++)
		//{
		//	objects->objects->label = new char[MAX_LABEL_LEN];
		//}
	}

	const DetectedObjects* ssd_detector::detect(unsigned char* data, int height, int width, float confidence_threshold)
	{
		OutputDebugPrintf("Enter detection dll.detect, height: %d, width: %d", height, width);
		if (data == NULL || height == 0 || width == 0)
			return NULL;

		//init cpu/gpu mode and device id
		if (!initialized) {
			if (Predictor::setDevice(gpuid))
			{
				initialized = true;
			}
		}

		cv::Mat img = Mat(height, width, CV_8UC3, data);
		//vector<int> caffeshape = ((Predictor *)myssd)->getInputBlobShape();
		//resize(img, img, Size(caffeshape[2], caffeshape[3]));

		long long start = milliseconds_now();
		((Predictor *)myssd)->Predict(img);
		long long end = milliseconds_now();
		cout << "Prediction time: " << (int)(end - start) << endl;

		vector<vector<float>> predictions;
		vector<vector<int>> shapes;
		vector<string> names = { "detection_out" };
		((Predictor *)myssd)->getBlobs(names, predictions, shapes);

		/* Copy the output layer to a std::vector */
		vector<float> prediction = predictions[0];
		const int num_det = shapes[0][2];
		vector<vector<float> > detections;
		int idx = 0;
		for (int k = 0; k < num_det; ++k) {
			if (prediction[idx] == -1) {
				// Skip invalid detection.
				idx += 7;
				continue;
			}
			vector<float> detection;
			for (int j = 0; j < 7; j++)
				detection.push_back(prediction[idx + j]);
			detections.push_back(detection);
			idx += 7;
		}

		objects->number = 0;

		for (int i = 0; i < detections.size(); ++i) {
			const vector<float>& d = detections[i];
			// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			if (d.size() != 7)
				OutputDebugPrintf("Error output of detection!");
			//CHECK_EQ(d.size(), 7);
			const float score = d[2];
			if (score >= confidence_threshold && objects->number < MAX_OBJ_COUNT) {
				DetectedObject* o = &(objects->objects[objects->number]);
				//o->label = new char[label_list[static_cast<int>(d[1])].length()];
				strcpy_s(o->label, (*((map<int, string>*)label_dic))[static_cast<int>(d[1])].c_str());
				o->score = score;
				o->xmin = static_cast<int>(d[3] * img.cols);
				o->ymin = static_cast<int>(d[4] * img.rows);
				o->xmax = static_cast<int>(d[5] * img.cols);
				o->ymax = static_cast<int>(d[6] * img.rows);
				objects->number++;
				OutputDebugPrintf("\tLabel: %d, Name: %s", static_cast<int>(d[1]), o->label);
			}
		}
		OutputDebugPrintf("Exit detection dll.detect, objects number: %d", objects->number);

		return objects;
	}


	void ssd_detector::detect_stop(void) {
		delete myssd;
		myssd = NULL;
		delete label_dic;
		label_dic = NULL;
		objects->number = 0;
		delete objects->objects;
		delete objects;
	}
}