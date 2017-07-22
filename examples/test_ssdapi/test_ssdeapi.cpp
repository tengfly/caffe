// TestCaffe.cpp : Defines the entry point for the console application.
//
#include <caffe/prediction.hpp>
#include <caffe/ssd_detection.hpp>
#include <caffe/include_symbols.hpp>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <iostream>
#include <iomanip>
using namespace caffe;
using namespace std;
using namespace cv;

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

int main() //_tmain(int argc, _TCHAR* argv[])
{
	string modelpath = "e:/code/caffe/modelzoo/model_ssd_coco300";
	ssd_detector ssddetector;
	ssddetector.detect_init(modelpath.c_str(), 3);

	string imagefile = "examples/images/heros.jpg";
	Mat img = imread(imagefile);
	const DetectedObjects* ssdresult;
	while (true)
	{
		long long start = milliseconds_now();
		ssdresult = ssddetector.detect(img.data, img.size().height, img.size().width, 0.50);
		long long end = milliseconds_now();

		cout << "Cost time: " << end - start << endl;
	}


	int number = ssdresult->number;
	//draw the bounding box into the original image
	for (int i = 0; i < number; i++)
	{
		DetectedObject obj = ssdresult->objects[i];
		cv::rectangle(img, cv::Rect(obj.xmin, obj.ymin, (obj.xmax - obj.xmin + 1), (obj.ymax - obj.ymin + 1)), cv::Scalar(0, 0, 255));
	}
	char winname[] = "SSD Detection";
	namedWindow(winname, CV_WINDOW_AUTOSIZE);
	imshow(winname, img);
	waitKey(0);


	ssddetector.detect_stop();
		
	return 0;
}

