#include <iostream>

#include <aruco/aruco.h>
#include <aruco/markerdetector.h>
#include <aruco/aruco_export.h>

#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;



int main(int argc, char** argv) {


	try {
		if(argc < 2) {
			std::cout<< "Usage: ./" << argv[0] << " [Camera Num]\n";
			return 1;
		}

		int camNum = atoi(argv[1]);

		aruco::MarkerDetector MDetector;
		vector<aruco::Marker> Markers;
		// Read the web cam

		Mat frame;
		VideoCapture cap;
		if(!cap.open(camNum, cv::CAP_ANY))
			cout << "Could not open camera!\n";
		while (cap.isOpened() && cap.read(frame))
		{
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