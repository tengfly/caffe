#pragma once
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <string>
#include <vector>

namespace caffe
{
	using std::string;
	/* Pair (label, confidence) representing a prediction. */
	typedef std::pair<string, float> Prediction;

	class Predictor {
	public:
		Predictor(const string& model_file,
			const string& trained_file,
			const string& mean_file,
			const string& label_file);
		~Predictor();

		std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
		bool Predict(const cv::Mat& img);
		const std::vector<int>& getInputBlobShape();
		bool getBlobs(const std::vector<std::string>& blobNames, std::vector< std::vector<float> >& blobdata, std::vector< std::vector<int> >& blobshape);
		void getBlobNames(std::vector<std::string>& blobNames);

		static bool setDevice(const int device_id);
	private:
		void SetMean(const string& mean_file);
		void WrapInputLayer(std::vector<cv::Mat>* input_channels);
		void Preprocess(const cv::Mat& img,
			std::vector<cv::Mat>* input_channels);

	private:
		void * net_;// shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Mat mean_;
		std::vector<string> labels_;
	};
}

