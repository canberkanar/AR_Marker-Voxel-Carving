#include <iostream>

#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <aruco/aruco.h>
#include <aruco/markerdetector.h>
#include <aruco/aruco_export.h>

#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";


int main(int argc, char** argv) {
	std::cout << "starting\n";

	try {
		std::cout << "starting2\n";

		CommandLineParser parser(argc, argv, params);
		parser.about("This program shows how to use background subtraction methods provided by "
			" OpenCV. You can process both videos and images.\n");
		if (parser.has("help"))
		{
			//print help information
			parser.printMessage();
		}

		//create Background Subtractor objects
		Ptr<BackgroundSubtractor> pBackSub;
		if (parser.get<String>("algo") == "MOG2") {
			pBackSub = createBackgroundSubtractorMOG2();
			std::cout << "using MOG2";
		}
		else {
			pBackSub = createBackgroundSubtractorKNN();
			std::cout << "using KNN";
		}

		if(argc < 2) {
			std::cout<< "Usage: ./" << argv[0] << " [Camera Num]\n";
			return 1;
		}

		int camNum = atoi(argv[1]);

		aruco::MarkerDetector MDetector;
		vector<aruco::Marker> Markers;
		// Read the web cam

		Mat frame;
		Mat fgMask;
		VideoCapture cap;
		if(!cap.open(camNum, cv::CAP_ANY))
			cout << "Could not open camera!\n";
		while (cap.isOpened() && cap.read(frame))
		{
			//update the background model
			pBackSub->apply(frame, fgMask);
			//get the frame number and write it on the current frame
			rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
				cv::Scalar(255, 255, 255), -1);
			stringstream ss;
			ss << cap.get(CAP_PROP_POS_FRAMES);
			string frameNumberString = ss.str();
			//putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
				//FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			//show the current frame and the fg masks
			imshow("FG Mask", fgMask);
			//get the input from the keyboard
			int keyboard = waitKey(30);
			if (keyboard == 'q' || keyboard == 27)
				break;
			// Detect markers in frame
			Markers = MDetector.detect(frame);

			//for each marker, draw info and its boundaries in the image

			for (auto& Marker : Markers) {

				cout << Marker << endl;

				Marker.draw(frame, Scalar(0, 0, 255), 2);

			}
			imshow("ocv", frame);
			if (waitKey(10) == 27) break;
		}
	}
	catch (std::exception& ex)
	{

		cout << "Exception :" << ex.what() << endl;

	}

}