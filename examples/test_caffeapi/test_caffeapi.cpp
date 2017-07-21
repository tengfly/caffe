// TestCaffe.cpp : Defines the entry point for the console application.
//

#include <caffe/prediction.hpp>
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
	string modelpath = "models\\bvlc_reference_caffenet";
	string model_proto = modelpath + "\\deploy.prototxt";
	string model_weight = modelpath + "\\model.caffemodel";
	string mean_file = "";
	string label_file = modelpath + "\\synset_words.txt";
	string imagefile = "examples\\images\\cat.jpg";
	Mat img = imread(imagefile);
	vector<string> blobnames;
	vector<vector<float>> blobdata;
	vector<vector<int>> blobshapes;
	Predictor caffert(model_proto, model_weight, mean_file, label_file);
	vector<int> caffeshape = caffert.getInputBlobShape();
	resize(img, img, Size(caffeshape[2], caffeshape[3]));
	//while (true)
	//{
	//	long long start = milliseconds_now();
	//	bool result = caffert.Predict(img);
	//	long long elapsed = milliseconds_now() - start;
	//	cout << "SSD cost time : " << elapsed << endl;
	//	if (result)
	//	{
	//		caffert.Classify(img, 5);
	//		//caffert.getBlobNames(blobnames);
	//		//caffert.getBlobs(blobnames, blobdata, blobshapes);	

	//		//cout << "Blob names:" <<endl;
	//		//for (int i = 0; i < blobnames.size(); i++)
	//		//	cout << blobnames[i] << endl;

	//	}
	//}

	vector<Prediction> predictions = caffert.Classify(img);

	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		cout << fixed << setprecision(4) << p.second << " - \""
			<< p.first << "\"" << endl;
	}

		
	return 0;
}


