#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
const char* params
		= "{ help h         |           | Print usage }"
		  "{ input          | vtest.avi | Path to a video or a sequence of image }"
		  "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
int main(int argc, char* argv[])
{

	if(argc < 2) {
		std::cout<< "Usage: ./" << argv[0] << " [Camera Num]\n";
		return 1;
	}

	int camNum = atoi(argv[1]);
	// Read the web cam
	VideoCapture cap;

	if (argc >= 4) {
		int dimX = atoi(argv[2]);
		int dimY = atoi(argv[3]);
		cap.set(cv::CAP_PROP_FRAME_WIDTH, dimX);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, dimY);
		if (argc == 5) {
			int fps = atoi(argv[4]);
			cap.set(cv::CAP_PROP_FPS, fps);
		}
	}

	//create Background Subtractor objects
	Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorKNN();
	Mat frame, fgMask;
	if(!cap.open(camNum, cv::CAP_ANY))
		cout << "Could not open camera!\n";
	while (cap.isOpened() && cap.read(frame)) {
		//update the background model
		pBackSub->apply(frame, fgMask);
		//get the frame number and write it on the current frame
		rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
				  cv::Scalar(255,255,255), -1);
		stringstream ss;
		ss << cap.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
				FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		if (waitKey(10) == 27) break;
	}
	return 0;
}