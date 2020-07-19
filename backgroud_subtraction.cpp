#include <iostream>
#include <filesystem>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

int main(int argc, char* argv[])
{

	if(argc < 2) {
		std::cout<< "Usage: ./" << argv[0] << " [Camera Num or File Path with format imgX.png]\n";
		return 1;
	}

	cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();

	if(std::filesystem::exists(argv[1])) {
		std::cout << "Interpreting input as File Path.\n";
		std::vector<std::string> images;
		cv::glob(std::string(argv[1]) + "img*.png", images);

		unsigned x = 0;

		for (const auto &file : images) {
			cv::Mat img = cv::imread(file);
			cv::Mat fgMask = subtractBackground(img, pBackSub);

			cv::imwrite(std::string(argv[1]) + "mask_" + std::to_string(x) + ".png", fgMask);
			x++;
		}
	} else {
		std::cout << "Interpreting input as camera number.\n";
		int camNum = std::atoi(argv[1]);
		// Read the web cam
		cv::VideoCapture cap;
		cv::Mat frame, fgMask;
		if(!cap.open(camNum, cv::CAP_ANY))
			std::cout << "Could not open camera!\n";
		while (cap.isOpened() && cap.read(frame)) {
			fgMask = subtractBackground(frame, pBackSub);
			//show the current frame and the fg masks
			cv::imshow("Frame", frame);
			cv::imshow("FG Mask", fgMask);
			if (cv::waitKey(10) == 27) break;
		}
	}
	return 0;
}