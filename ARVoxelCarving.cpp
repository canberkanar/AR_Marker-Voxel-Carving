#include <iostream>

#include <aruco/aruco.h>
#include <aruco/markerdetector.h>
#include <aruco/aruco_export.h>

#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;



int main(int argc, char** argv) {


	try {

		aruco::MarkerDetector MDetector;

		vector<aruco::Marker> Markers;

		// Read the web cam

		Mat frame;
		VideoCapture cap;
		cap.open(1, cv::CAP_DSHOW);
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